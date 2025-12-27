import logging
import os
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
import json
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
logger = logging.getLogger(__name__)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


langfuse_handler = CallbackHandler()
langfuse = Langfuse()


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  
    search_steps: int
    max_search_steps: int
    retrieve_results: RetrievalResult


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

        if retrieval_mode == "dense":
            res = retrieve_dense(query, chroma_index, embedding_client, top_k=k)

        elif retrieval_mode == "keyword":
            if bm25_index is None:
                raise ValueError("BM25Index required for keyword retrieval")
            res = retrieve_keyword(query, bm25_index, top_k=k)

        elif retrieval_mode == "hybrid":
            if bm25_index is None:
                raise ValueError("BM25Index required for hybrid retrieval")
            res = retrieve_hybrid(query, chroma_index, embedding_client, bm25_index, top_k=k)

        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")

        if reranker is not None:
            res = apply_reranker(res, query, reranker, top_k=k)

        return res.to_dict()

    return retrieve


class AgenticRAGGraph:
    def __init__(
        self,
        *,
        model: str,
        chroma_index,
        embedding_client,
        bm25_index=None,
        retrieval_mode: str = "dense",
        reranker: Optional[Any] = None,
        temperature: float = 0.2,
        max_search_steps: int = 3,
    ):
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")

        self.llm = ChatOpenAI(
            model=model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
        )


        self.retrieval_mode = retrieval_mode
        self.chroma_index = chroma_index
        self.embedding_client = embedding_client
        self.bm25_index = bm25_index
        self.reranker = reranker

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

        self.system_prompt = SystemMessage(
            content=(
                "You are a RAG agent.\n"
                "You have access to a retrieve tool that returns context chunks.\n\n"
                "Search strategy:\n"
                f"- You may call retrieve multiple times (query refinement allowed), up to {max_search_steps} total searches.\n"
                "- After each retrieve, read the returned context. If it’s insufficient, refine the query and retrieve again.\n"
                "- When you have enough info OR when the search budget is exhausted, write the final answer.\n"
                "- If the budget is exhausted, DO NOT request more tools; answer using only what you already have.\n"
            )
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
            input_msgs = [self.system_prompt] + msgs

        ai = self.llm_with_tools.invoke(input_msgs)
        tool_calls = getattr(ai, "tool_calls", None) or []


        return {
            "messages": [ai],
        }

    def _route_after_llm(self, state: AgentState) -> Literal["tools", "final"]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        if tool_calls:
            return "tools"
        else:

            state["search_steps"] = 0
            return "final"


    def _handle_tools(self, state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        for tc in tool_calls:
            if tc.get("name") == "retrieve":

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


                from ragbench.retrievers.retrieval import (
                    retrieve_dense,
                    retrieve_keyword,
                    retrieve_hybrid,
                    apply_reranker,
                )

                if self.retrieval_mode == "dense":
                    tool_result = retrieve_dense(query, self.chroma_index, self.embedding_client, top_k=k).to_dict()
                elif self.retrieval_mode == "keyword":
                    if self.bm25_index is None:
                        raise ValueError("BM25Index required for keyword retrieval")
                    tool_result = retrieve_keyword(query, self.bm25_index, top_k=k).to_dict()
                elif self.retrieval_mode == "hybrid":
                    if self.bm25_index is None:
                        raise ValueError("BM25Index required for hybrid retrieval")
                    tool_result = retrieve_hybrid(query, self.chroma_index, self.embedding_client, self.bm25_index, top_k=k).to_dict()
                else:
                    raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

                if self.reranker is not None:

                    from ragbench.retrievers.base import RetrievalResult
                    result_obj = RetrievalResult.from_dict(tool_result)
                    result_obj = apply_reranker(result_obj, query, self.reranker, top_k=k)
                    tool_result = result_obj.to_dict()

                content = json.dumps(tool_result)
                tool_message = ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    content=content,
                )


                new_search_steps = state["search_steps"] + 1

                return {"messages": [tool_message], "retrieve_results": tool_result, "search_steps": new_search_steps}
            else:   
                logger.warning(f"Unknown tool call: {tc.get('name')}")
                tool_message=ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    content=json.dumps({"error": f"Unknown tool call: {tc.get('name')}"}),
                )
                return {"messages": [tool_message], "retrieve_results": {}}

    def run(self, query: str, max_search_steps: int =3) -> dict:
        """Run Agentic RAG pipeline with Langfuse tracing."""
        # Run the graph with Langfuse tracing (callbacks already configured)
        state = self.graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "search_steps": 0,
                "max_search_steps": max_search_steps
            }
        )

        final_message = state["messages"][-1]
        response = getattr(final_message, "content", "")
        context = ""
        retrieval_results = []
        total_chunks = 0
        total_latency = 0

        for msg in state["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "retrieve":
                        pass
            elif hasattr(msg, "tool_call_id"):
                try:
                    tool_result = json.loads(msg.content)
                    # Handle both "chunks" (old format) and "documents" (new to_dict format)
                    if "documents" in tool_result:
                        chunks_text = tool_result["documents"]
                        context += "\n".join(chunks_text) + "\n"
                        retrieval_results.append(tool_result)
                        total_chunks += len(chunks_text)
                        total_latency += tool_result.get("latency_ms", 0)
                    elif "chunks" in tool_result:
                        chunks_text = [chunk.get("text", "") for chunk in tool_result["chunks"]]
                        context += "\n".join(chunks_text) + "\n"
                        retrieval_results.append(tool_result)
                        total_chunks += len(tool_result["chunks"])
                        total_latency += tool_result.get("latency_ms", 0)
                except Exception as e:
                    logger.debug(f"Failed to parse tool result: {e}")

        # Log custom metrics to Langfuse using best practices
        self._log_performance_metrics(state, total_chunks, total_latency)

        return {
            "response": response,
            "context": context.strip(),
            "retrieval_result": retrieval_results[-1] if retrieval_results else {},
            "messages": state["messages"],
            "search_steps": state["search_steps"],
        }

    def _log_performance_metrics(self, state: dict, total_chunks: int, total_latency: float):
        """Log performance metrics to Langfuse following best practices."""
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
