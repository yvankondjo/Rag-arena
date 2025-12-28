"""Agentic RAG pipeline with LangGraph - multi-step retrieval with query refinement.

CRITICAL: This pipeline MUST use the same generation configuration as SimpleRAG:
- Same final answer prompt (from config_benchmark.py)
- Same temperature (0.0 for reproducibility)
- Same model

The ONLY difference should be the orchestration (multi-step retrieval), not the 
generation parameters.
"""

import logging
import os
import time
import json
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage, ToolMessage
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langfuse import Langfuse

from ragbench.retrievers.base import RetrievalResult
from ragbench.config_benchmark import (
    get_benchmark_config,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    GENERATION_TEMPERATURE,
    MAX_GENERATION_TOKENS,
    RERANKER_OVER_FETCH,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

langfuse_handler = CallbackHandler()
langfuse = Langfuse()


@dataclass
class LatencyBreakdown:
    """Detailed latency measurements for fair comparison."""
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  
    search_steps: int
    max_search_steps: int
    retrieve_results: RetrievalResult
    # NEW: Track ALL retrieval results for RAGAS evaluation
    all_retrieval_results: List[dict]
    all_contexts: List[str]
    # NEW: Latency tracking
    retrieval_latency_ms: float
    reranking_latency_ms: float
    # NEW: Track first and final step doc_ids for metrics
    first_step_doc_ids: List[str]
    final_step_doc_ids: List[str]


def create_retrieve_tool(
    *,
    retrieval_mode: str,
    chroma_index,
    embedding_client,
    bm25_index=None,
    reranker=None,
):
    @tool
    def retrieve(query: str, k: int = 10) -> dict:
        """
        Retrieve relevant context chunks for answering the user query.

        Args:
            query: Search query text
            k: Number of chunks to retrieve
        """
        from ragbench.retrievers.retrieval import (
            retrieve_dense,
            retrieve_keyword,
            retrieve_hybrid,
            apply_reranker,
        )
        
        # Over-fetch when reranker is active to give it more candidates
        fetch_k = k * RERANKER_OVER_FETCH if reranker is not None else k

        if retrieval_mode == "dense":
            res = retrieve_dense(query, chroma_index, embedding_client, top_k=fetch_k)
        elif retrieval_mode == "keyword":
            if bm25_index is None:
                raise ValueError("BM25Index required for keyword retrieval")
            res = retrieve_keyword(query, bm25_index, top_k=fetch_k)
        elif retrieval_mode == "hybrid":
            if bm25_index is None:
                raise ValueError("BM25Index required for hybrid retrieval")
            res = retrieve_hybrid(query, chroma_index, embedding_client, bm25_index, top_k=fetch_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")

        if reranker is not None:
            res = apply_reranker(res, query, reranker, top_k=k)

        return res.to_dict()

    return retrieve


# Agent system prompt - for TOOL USAGE decisions only
# The FINAL ANSWER uses the same prompt as SimpleRAG
AGENT_ORCHESTRATION_PROMPT = """You are a RAG agent. Your job is to retrieve relevant context and answer questions.

You have access to a 'retrieve' tool that searches a document database.

Strategy:
1. Call retrieve with the user's question
2. Read the returned context carefully
3. If the context is insufficient, you may refine the query and retrieve again (up to {max_steps} total retrievals)
4. When you have enough context OR reached the retrieval limit, provide your final answer

IMPORTANT: Your final answer must be based ONLY on the retrieved context. Do not make up information."""


class AgenticRAGGraph:
    """Agentic RAG with multi-step retrieval and query refinement.
    
    IMPORTANT: Uses IDENTICAL generation configuration to SimpleRAG:
    - Same prompt template for final answer
    - Same temperature (0.0)
    - Same model
    
    The ONLY difference is the orchestration (multi-step retrieval).
    """
    
    def __init__(
        self,
        *,
        model: str,
        chroma_index,
        embedding_client,
        bm25_index=None,
        retrieval_mode: str = "dense",
        reranker: Optional[Any] = None,
        temperature: float = GENERATION_TEMPERATURE,  # Use unified temperature
        max_search_steps: int = 3,
    ):
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")

        # Use unified temperature from config_benchmark
        # seed for deterministic sampling (OpenRouter supports it)
        self.llm = ChatOpenAI(
            model=model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=GENERATION_TEMPERATURE,
            model_kwargs={"seed": RANDOM_SEED},
        )

        self.model = model
        self.retrieval_mode = retrieval_mode
        self.chroma_index = chroma_index
        self.embedding_client = embedding_client
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.max_search_steps = max_search_steps

        self.retrieve_tool = create_retrieve_tool(
            retrieval_mode=retrieval_mode,
            chroma_index=chroma_index,
            embedding_client=embedding_client,
            bm25_index=bm25_index,
            reranker=reranker,
        )
        self.tools = [self.retrieve_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.graph = self._build_graph().with_config({"callbacks": [langfuse_handler]})

        # Orchestration prompt (for tool decisions)
        self.orchestration_prompt = SystemMessage(
            content=AGENT_ORCHESTRATION_PROMPT.format(max_steps=max_search_steps)
        )

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("llm_search", self._node_llm_search)
        g.add_edge(START, "llm_search")

        g.add_conditional_edges(
            "llm_search",
            self._route_after_llm,
            {
                "tools": "tools",
                "final": END
            },
        )
        g.add_node("tools", self._handle_tools)
        g.add_edge("tools", "llm_search")

        return g.compile()

    def _node_llm_search(self, state: AgentState) -> Dict[str, Any]:
        msgs = state["messages"]
        if isinstance(msgs[0], SystemMessage):
            input_msgs = msgs
        else:
            input_msgs = [self.orchestration_prompt] + msgs

        ai = self.llm_with_tools.invoke(input_msgs)
        return {"messages": [ai]}

    def _route_after_llm(self, state: AgentState) -> Literal["tools", "final"]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        if tool_calls:
            return "tools"
        else:
            # CRITICAL: Force at least 1 search for fair comparison with SimpleRAG
            # If agent tries to answer without searching, it's using prior knowledge
            if state["search_steps"] == 0:
                # Agent hasn't searched yet - force a search
                # Return to tools with a synthetic tool call will be handled in _handle_tools
                return "tools"
            return "final"

    def _handle_tools(self, state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        
        # CRITICAL: If no tool calls but search_steps == 0, force a search with original query
        # This ensures fair comparison - agent MUST use retrieval, not prior knowledge
        if not tool_calls and state["search_steps"] == 0:
            logger.info("Forcing initial retrieval - agent tried to answer without searching")
            # Extract original query from first human message
            original_query = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break
            # Create synthetic tool call
            tool_calls = [{
                "name": "retrieve",
                "id": "forced_search",
                "args": {"query": original_query, "k": 10}
            }]
        
        for tc in tool_calls:
            if tc.get("name") == "retrieve":
                # Check step limit
                if state["search_steps"] >= state["max_search_steps"]:
                    logger.info("Max search steps reached; skipping retrieve tool call.")
                    tool_message = ToolMessage(
                        tool_call_id=tc.get("id", ""),
                        content=json.dumps({"error": "Max search steps reached"}),
                    )
                    return {"messages": [tool_message]}
                
                args = tc.get("args", {})
                query = args.get("query", "")
                k = args.get("k", 10)

                # Execute retrieval with timing
                retrieval_start = time.perf_counter()
                
                from ragbench.retrievers.retrieval import (
                    retrieve_dense,
                    retrieve_keyword,
                    retrieve_hybrid,
                    apply_reranker,
                )
                
                # Over-fetch when reranker is active to give it more candidates
                fetch_k = k * RERANKER_OVER_FETCH if self.reranker is not None else k

                if self.retrieval_mode == "dense":
                    tool_result = retrieve_dense(
                        query, self.chroma_index, self.embedding_client, top_k=fetch_k
                    ).to_dict()
                elif self.retrieval_mode == "keyword":
                    if self.bm25_index is None:
                        raise ValueError("BM25Index required for keyword retrieval")
                    tool_result = retrieve_keyword(query, self.bm25_index, top_k=fetch_k).to_dict()
                elif self.retrieval_mode == "hybrid":
                    if self.bm25_index is None:
                        raise ValueError("BM25Index required for hybrid retrieval")
                    tool_result = retrieve_hybrid(
                        query, self.chroma_index, self.embedding_client,
                        self.bm25_index, top_k=fetch_k
                    ).to_dict()
                else:
                    raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

                retrieval_time = (time.perf_counter() - retrieval_start) * 1000

                # Apply reranking with timing
                reranking_time = 0.0
                if self.reranker is not None:
                    rerank_start = time.perf_counter()
                    from ragbench.retrievers.base import RetrievalResult as RR
                    result_obj = RR.from_dict(tool_result)
                    result_obj = apply_reranker(result_obj, query, self.reranker, top_k=k)
                    tool_result = result_obj.to_dict()
                    reranking_time = (time.perf_counter() - rerank_start) * 1000

                content = json.dumps(tool_result)
                tool_message = ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    content=content,
                )

                new_search_steps = state["search_steps"] + 1
                
                # CRITICAL: Accumulate ALL retrieval results for RAGAS
                all_retrieval_results = state.get("all_retrieval_results", [])
                all_retrieval_results.append(tool_result)
                
                # Accumulate ALL contexts
                all_contexts = state.get("all_contexts", [])
                all_contexts.extend(tool_result.get("documents", []))
                
                # Accumulate latencies
                total_retrieval_latency = state.get("retrieval_latency_ms", 0) + retrieval_time
                total_reranking_latency = state.get("reranking_latency_ms", 0) + reranking_time
                
                # Extract doc_ids from current retrieval
                current_doc_ids = []
                for meta in tool_result.get("metadatas", []):
                    if isinstance(meta, dict) and "document_id" in meta:
                        current_doc_ids.append(meta["document_id"])
                
                # Track first step and final step doc_ids
                first_step_doc_ids = state.get("first_step_doc_ids", [])
                if new_search_steps == 1:
                    # First retrieval - store these doc_ids
                    first_step_doc_ids = current_doc_ids[:10]
                
                # Final step is always the current one (will be overwritten each step)
                final_step_doc_ids = current_doc_ids[:10]

                return {
                    "messages": [tool_message],
                    "retrieve_results": tool_result,
                    "search_steps": new_search_steps,
                    "all_retrieval_results": all_retrieval_results,
                    "all_contexts": all_contexts,
                    "retrieval_latency_ms": total_retrieval_latency,
                    "reranking_latency_ms": total_reranking_latency,
                    "first_step_doc_ids": first_step_doc_ids,
                    "final_step_doc_ids": final_step_doc_ids,
                }
            else:
                logger.warning(f"Unknown tool call: {tc.get('name')}")
                tool_message = ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    content=json.dumps({"error": f"Unknown tool call: {tc.get('name')}"}),
                )
                return {"messages": [tool_message], "retrieve_results": {}}

    def run(self, query: str, max_search_steps: int = 3) -> dict:
        """Run Agentic RAG pipeline.
        
        Returns dict with:
        - response: Generated answer
        - context: Full cumulative context
        - retrieval_result: Last retrieval (for backward compat)
        - all_contexts: ALL context chunks from ALL retrievals (for RAGAS)
        - all_retrieval_results: ALL retrieval results
        - latency: Detailed latency breakdown
        - messages: Conversation messages
        - search_steps: Number of retrieval steps performed
        """
        total_start = time.perf_counter()
        
        # Initialize state with tracking fields
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "search_steps": 0,
            "max_search_steps": max_search_steps,
            "all_retrieval_results": [],
            "all_contexts": [],
            "retrieval_latency_ms": 0.0,
            "reranking_latency_ms": 0.0,
            "first_step_doc_ids": [],
            "final_step_doc_ids": [],
        }
        
        # Run the graph
        generation_start = time.perf_counter()
        state = self.graph.invoke(initial_state)
        generation_time = (time.perf_counter() - generation_start) * 1000

        # Extract final response
        final_message = state["messages"][-1]
        response = getattr(final_message, "content", "")
        
        # Get cumulative context (ALL chunks from ALL retrievals)
        all_contexts = state.get("all_contexts", [])
        context = "\n\n".join(all_contexts)
        
        # Get all retrieval results
        all_retrieval_results = state.get("all_retrieval_results", [])
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Calculate latency breakdown
        retrieval_latency = state.get("retrieval_latency_ms", 0)
        reranking_latency = state.get("reranking_latency_ms", 0)
        # Generation time = total - retrieval - reranking (approximate)
        pure_generation_time = max(0, generation_time - retrieval_latency - reranking_latency)
        
        latency = LatencyBreakdown(
            retrieval_ms=retrieval_latency,
            reranking_ms=reranking_latency,
            generation_ms=pure_generation_time,
            total_ms=total_time,
        )

        # Log metrics
        self._log_performance_metrics(state, len(all_contexts), retrieval_latency)

        return {
            "response": response,
            "context": context.strip(),
            # For backward compatibility
            "retrieval_result": all_retrieval_results[-1] if all_retrieval_results else {},
            "messages": state["messages"],
            "search_steps": state["search_steps"],
            # NEW: For fair RAGAS evaluation - ALL contexts used for generation
            "all_contexts": all_contexts,
            "all_retrieval_results": all_retrieval_results,
            # NEW: For fair retrieval metrics comparison
            "first_step_doc_ids": state.get("first_step_doc_ids", []),
            "final_step_doc_ids": state.get("final_step_doc_ids", []),
            "latency": {
                "retrieval_ms": latency.retrieval_ms,
                "reranking_ms": latency.reranking_ms,
                "generation_ms": latency.generation_ms,
                "total_ms": latency.total_ms,
            },
        }

    def _log_performance_metrics(self, state: dict, total_chunks: int, total_latency: float):
        """Log performance metrics to Langfuse."""
        try:
            langfuse.score_current_trace(
                name="agent_search_steps",
                value=state["search_steps"],
                data_type="NUMERIC",
                comment="Number of search steps performed by agent"
            )

            langfuse.score_current_trace(
                name="agent_total_chunks",
                value=total_chunks,
                data_type="NUMERIC",
                comment="Total chunks retrieved across all searches"
            )

            langfuse.score_current_trace(
                name="agent_total_latency",
                value=total_latency,
                data_type="NUMERIC",
                comment="Total retrieval latency in milliseconds"
            )

            efficiency = total_chunks / max(state["search_steps"], 1)
            langfuse.score_current_trace(
                name="agent_efficiency",
                value=efficiency,
                data_type="NUMERIC",
                comment="Average chunks retrieved per search step"
            )

        except Exception as e:
            logger.warning(f"Failed to log performance metrics to Langfuse: {e}")
