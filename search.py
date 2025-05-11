import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json
import re
from typing import List, Dict, Any, Optional

# Import LangChain modules
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain.chains.base import Chain


# Function 1: Initialize model and vector storage
def initialize_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(384)  # Vector dimension
    return model, index


# Function 2: Fetch news content using NewsAPI
def fetch_news_content(query, api_key="96ba4ddece1040948eea2819715806c2"):
    """Fetch news from NewsAPI.org using the provided API key"""
    try:
        # Clean and validate query content
        clean_query = query.strip()

        # Check if it contains tags or text that's too long
        if '<' in clean_query or '>' in clean_query or len(clean_query) > 500:
            # Remove all XML/HTML tags
            clean_query = re.sub(r'<.*?>', '', clean_query)
            # Remove excess whitespace
            clean_query = re.sub(r'\s+', ' ', clean_query).strip()

            # If still too long, only take the first 100 characters
            if len(clean_query) > 100:
                words = clean_query.split()
                if len(words) > 5:
                    clean_query = ' '.join(words[:5])
                else:
                    clean_query = clean_query[:100]

        print(f"Clean search query: {clean_query}")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": clean_query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": api_key
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        news_data = response.json()

        print(f"Found {len(news_data.get('articles', []))} articles about '{clean_query}'")
        return news_data

    except Exception as e:
        print(f"Error fetching news from API: {e}")
        # Return empty structure in case of error
        return {"articles": []}


# Function 3: Get response from local models (either Qwen or DeepSeek)
def get_model_response(prompt_text, model_type="qwen", max_tokens=500):
    """
    Get response from a local model

    Parameters:
    - prompt_text: The text prompt to send to the model
    - model_type: 'qwen' or 'deepseek' to specify which model to use
    - max_tokens: Maximum tokens to generate

    Returns:
    - Model response as string
    """
    try:
        # Extract query from prompt text for fallback keywords if needed
        query_match = re.search(r'User query: (.*?)(?:\n|$)', prompt_text)
        fallback_query = query_match.group(1) if query_match else "current news"

        if model_type.lower() == "qwen":
            # Use Qwen via Ollama for keyword extraction
            enhanced_prompt = prompt_text + "\n\nIMPORTANT: Provide only the keywords without any explanations or additional text. Just list 1-3 main keywords separated by commas."

            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "qwen3:0.6b",
                "prompt": enhanced_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Lower temperature for more deterministic output
                "stream": False
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                response_json = response.json()
                raw_text = response_json.get("response", "")

                # Clean output, remove thinking process
                clean_text = raw_text

                # Remove <think>...</think> blocks
                clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)

                # Remove other tags
                clean_text = re.sub(r'<.*?>', '', clean_text)

                # Remove common conversational prefixes
                common_prefixes = ["okay", "sure", "here", "certainly", "alright", "the keywords", "keywords",
                                   "keyword"]
                for prefix in common_prefixes:
                    pattern = rf"^{prefix}[,:\s]+"
                    clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)

                # Remove any extra explanations
                if ":" in clean_text:
                    clean_text = clean_text.split(":", 1)[1].strip()

                # Get rid of quotes if present
                clean_text = clean_text.replace('"', '').replace("'", "")

                # Remove any bullet points or numbering
                clean_text = re.sub(r"^\d+\.\s*", "", clean_text)
                clean_text = re.sub(r"^-\s*", "", clean_text)

                # Clean excess whitespace and line breaks
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                print(f"Qwen keywords: {clean_text}")
                return clean_text

            print(f"Error from Ollama API: Status {response.status_code} - {response.text[:100]}")
            # Extract fallback keywords from the query
            words = fallback_query.split()
            # Remove common stop words
            stop_words = {"is", "the", "a", "an", "in", "on", "at", "by", "for", "with", "about", "from", "to", "of"}
            keywords = [word for word in words if word.lower() not in stop_words]
            if not keywords:
                keywords = words[:3]
            fallback_keywords = ', '.join(keywords[:min(3, len(keywords))])
            print(f"Failed to get keywords from Qwen, using fallback: {fallback_keywords}")
            return fallback_keywords

        elif model_type.lower() == "deepseek":
            # Use DeepSeek for analysis
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "deepseek-r1:7b",
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Slightly higher for analysis
                "stream": False
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                full_text = ""
                for line in lines:
                    if line:
                        try:
                            data = json.loads(line)
                            full_text += data.get("response", data.get("text", ""))
                        except json.JSONDecodeError:
                            pass

                # Clean output
                clean_text = full_text.strip()
                # Remove any HTML/XML tags
                clean_text = re.sub(r'<.*?>', '', clean_text)

                return clean_text

            print(f"Error from Ollama API: Status {response.status_code} - {response.text[:100]}")
            return "Error occurred while analyzing the news articles."
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    except Exception as e:
        print(f"Error getting {model_type} response: {e}")
        if model_type.lower() == "qwen":
            # Extract default keywords from the query
            words = fallback_query.split()
            stop_words = {"is", "the", "a", "an", "in", "on", "at", "by", "for", "with", "about"}
            keywords = [word for word in words if word.lower() not in stop_words]
            if not keywords:
                keywords = words[:3]
            fallback_keywords = ', '.join(keywords[:min(3, len(keywords))])
            return fallback_keywords
        return "Error occurred while analyzing the news articles."


# Function 4: Create LLM and runnable chains
def create_runnable_chains():
    # Generic Model LLM class compatible with LangChain
    class ModelLLM(LLM):
        model_type: str = "qwen"
        max_tokens: int = 500

        @property
        def _llm_type(self) -> str:
            return f"{self.model_type}-local"

        def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs
        ) -> str:
            return get_model_response(prompt, self.model_type, self.max_tokens)

    # News search chain - using Qwen
    search_template = """
    You are a professional news analyst. Based on the user's query, identify relevant keywords for a news search.

    User query: {query}

    Return 1-3 specific keywords that would be most effective for searching news about this topic.
    Format as a simple comma-separated list, e.g.: "keyword1, keyword2, keyword3"

    Return only the keywords without any additional text.
    """

    search_prompt = PromptTemplate(
        input_variables=["query"],
        template=search_template
    )

    qwen_llm = ModelLLM(model_type="qwen", max_tokens=100)
    news_search_chain = search_prompt | qwen_llm | StrOutputParser()

    # News analysis chain - using DeepSeek
    analysis_template = """
    You are a professional news analyst and fact-checker. I will provide you with a user query and several news articles. Your task is to analyze these articles collectively and provide a direct answer to the user's question.

    User query: {query}

    Here are the articles I found:
    {article_summaries}

    Based on these articles, please provide:
    1. A direct answer to the user's question in one or two sentences
    2. A brief summary of the evidence from the articles supporting your answer
    3. An assessment of whether there's any conflicting information
    4. A confidence level in your answer (high/medium/low)

    Keep your answer concise, clear, and focused on answering the user's question. Cite specific articles when relevant.
    """

    analysis_prompt = PromptTemplate(
        input_variables=["query", "article_summaries"],
        template=analysis_template
    )

    deepseek_llm = ModelLLM(model_type="deepseek", max_tokens=800)
    news_analysis_chain = analysis_prompt | deepseek_llm | StrOutputParser()

    # Custom content fetcher chain with proper type annotations
    class ContentFetcherChain(Chain):
        # Add proper type annotations to class variables
        model: Optional[Any] = None
        index: Optional[Any] = None
        search_chain: Optional[Any] = None

        @property
        def input_keys(self) -> List[str]:
            return ["query"]

        @property
        def output_keys(self) -> List[str]:
            return ["articles", "search_query", "article_summaries"]

        def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = inputs["query"]

            # Get search keywords from Qwen
            try:
                search_keywords = self.search_chain.invoke({"query": query})
                # Clean up the keywords
                search_keywords = search_keywords.strip()
                if not search_keywords:
                    # Fallback to simple extraction if nothing returned
                    words = query.split()
                    stop_words = {"is", "the", "a", "an", "in", "on", "at", "by", "for", "with", "about"}
                    keywords = [word for word in words if word.lower() not in stop_words]
                    if not keywords:
                        keywords = words[:3]
                    search_keywords = ', '.join(keywords[:min(3, len(keywords))])
            except Exception as e:
                print(f"Error getting keywords from Qwen: {e}")
                # Fallback to simple extraction
                words = query.split()
                stop_words = {"is", "the", "a", "an", "in", "on", "at", "by", "for", "with", "about"}
                keywords = [word for word in words if word.lower() not in stop_words]
                if not keywords:
                    keywords = words[:3]
                search_keywords = ', '.join(keywords[:min(3, len(keywords))])

            print(f"Searching news for: {search_keywords}")

            # Fetch news content using NewsAPI
            news_data = fetch_news_content(search_keywords)
            articles = news_data.get("articles", [])

            # Process and index the articles if needed
            if self.model and self.index and articles:
                article_contents = [a.get("content", "") or a.get("description", "") for a in articles if
                                    a.get("content") or a.get("description")]
                if article_contents:
                    try:
                        vectors = self.model.encode(article_contents)
                        self.index.add(np.array(vectors))
                    except Exception as e:
                        print(f"Error indexing articles: {e}")

            # Create a summary of all articles for the analysis chain
            article_summaries = ""
            for i, article in enumerate(articles, 1):
                source = article.get("source", {}).get("name", "Unknown Source")
                title = article.get("title", "No Title")
                url = article.get("url", "")
                content = article.get("content", "") or article.get("description", "No content available")

                # Truncate content if too long
                content_snippet = content[:400] + "..." if len(content) > 400 else content

                # Add to the summaries
                article_summaries += f"ARTICLE {i}:\nSource: {source}\nTitle: {title}\nURL: {url}\nSummary: {content_snippet}\n\n"

            return {
                "articles": articles,
                "search_query": search_keywords,
                "article_summaries": article_summaries
            }

    # News analyzer chain with proper type annotations
    class NewsAnalyzerChain(Chain):
        analysis_chain: Optional[Any] = None

        @property
        def input_keys(self) -> List[str]:
            return ["query", "article_summaries"]

        @property
        def output_keys(self) -> List[str]:
            return ["analysis"]

        def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = inputs["query"]
            article_summaries = inputs["article_summaries"]

            if not article_summaries.strip():
                return {"analysis": "No relevant news articles were found to answer your question."}

            try:
                print("\nGenerating comprehensive analysis using DeepSeek...")

                # Get analysis from DeepSeek
                analysis = self.analysis_chain.invoke({
                    "query": query,
                    "article_summaries": article_summaries
                })

                return {"analysis": analysis}
            except Exception as e:
                print(f"Error during analysis: {e}")
                return {"analysis": f"Error analyzing the news articles: {str(e)[:100]}"}

    # Initialize model and index
    model, index = initialize_system()

    # Create chains
    content_fetcher_chain = ContentFetcherChain()
    analyzer_chain = NewsAnalyzerChain()

    # Connect chains
    content_fetcher_chain.model = model
    content_fetcher_chain.index = index
    content_fetcher_chain.search_chain = news_search_chain
    analyzer_chain.analysis_chain = news_analysis_chain

    return content_fetcher_chain, analyzer_chain


# Function 5: Process query and return results
def process_query(query):
    try:
        # Create chains for each query (this ensures fresh state)
        content_fetcher_chain, analyzer_chain = create_runnable_chains()

        # First fetch content using Qwen for keywords
        fetch_result = content_fetcher_chain.invoke({"query": query})

        # Then get comprehensive analysis using DeepSeek
        analysis_result = analyzer_chain.invoke({
            "query": query,
            "article_summaries": fetch_result["article_summaries"]
        })

        # Combine results
        return {
            "search_query": fetch_result["search_query"],
            "articles": fetch_result["articles"],
            "analysis": analysis_result["analysis"]
        }
    except Exception as e:
        print(f"Error in process_query: {e}")
        # Return minimal valid structure with the error
        return {
            "search_query": query,  # Use original query as fallback
            "articles": [],
            "analysis": f"Error processing query: {str(e)}"
        }


# Function 6: Main function
def main():
    # Get user query
    query = input("Please enter your query: ")

    # Process query
    results = process_query(query)

    # Print results
    print("\n\nNews Analysis Results:")
    print(f"Query: {query}")
    print(f"Search keywords: {results['search_query']}")
    print(f"Found {len(results['articles'])} relevant articles")

    if not results['articles']:
        print("No news articles were found matching your query.")
        return

    # Print the comprehensive analysis
    print("\nANALYSIS:")
    print(results['analysis'])

    # Print sources for reference
    print("\nSOURCES:")
    for i, article in enumerate(results['articles'], 1):
        print(f"{i}. {article.get('title', 'No Title')} - {article.get('source', {}).get('name', 'Unknown')}")
        print(f"   URL: {article.get('url', 'No URL')}")


if __name__ == "__main__":
    main()