"""Gradio UI interface - calls RAG services directly (in-process)"""
import gradio as gr
import time
from typing import Tuple
import logging

logger = logging.getLogger('rag_system.ui')

# Will be set by main.py
query_handler = None

# Import metrics collector from API routes (shared instance)
from app.api.routes import metrics


def set_query_handler(handler):
    """Set the query handler instance"""
    global query_handler
    query_handler = handler
    logger.info("Query handler set for Gradio UI")


def get_custom_css(color: str = "orange") -> str:
    """Generate custom CSS for Gradio interface with specified color theme"""
    return f"""
    .gradio-container {{
        max-width: 1400px !important;
    }}
    
    /* {color.capitalize()} accent color for sliders and buttons */
    input[type="range"]::-webkit-slider-thumb {{
        background: {color} !important;
    }}
    
    input[type="range"]::-moz-range-thumb {{
        background: {color} !important;
    }}
    
    .primary {{
        background: {color} !important;
        border-color: {color} !important;
    }}
    
    .primary:hover {{
        background: {color} !important;
    }}
    
    /* Dark theme adjustments */
    .dark .primary {{
        background: {color} !important;
    }}
    """


def query_rag(
    query: str,
    k: int,
    temperature: float,
    model: str
) -> Tuple[str, str]:
    """Process query using RAG (calls service directly, not HTTP)"""
    start_time = time.time()
    success = False
    error_type = None
    result = None
    
    if query_handler is None:
        return "❌ Error: RAG system not initialized", ""
    
    if not query or query.strip() == "":
        return "⚠️ Please enter a question", ""
    
    try:
        logger.info(f"UI Query: {query[:100]}...")
        
        # Call the query handler DIRECTLY (in-process, no HTTP)
        result = query_handler.answer_query(
            query=query,
            k=int(k),
            temperature=temperature,
            model=model
        )
        
        # Format answer
        answer = result["answer"]
        
        # Format retrieval metadata
        metadata = f"### 📊 Retrieval Info\n\n"
        metadata += f"**Retrieved {len(result['retrieval_results'])} results** (k={result['k']})\n\n"
        
        for r in result['retrieval_results'][:5]:
            metadata += f"**{r['rank']}.** {r['branch']} | Rating: {r['rating']}/5 | Distance: {r['distance']:.4f}\n"
            metadata += f"   *{r['snippet'][:150]}...*\n\n"
        
        success = True
        return answer, metadata
        
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Error in UI: {str(e)}", exc_info=True)
        return f"❌ Error: {str(e)}", ""
    
    finally:
        # Track metrics for UI requests
        response_time = time.time() - start_time
        query_length = len(query)
        
        # Extract additional metrics from result if available
        avg_distance = None
        response_length = None
        
        if result and success:
            # Calculate average retrieval distance
            if 'retrieval_results' in result and result['retrieval_results']:
                distances = [r['distance'] for r in result['retrieval_results']]
                avg_distance = sum(distances) / len(distances)
            
            # Get response length
            if 'answer' in result:
                response_length = len(result['answer'])
        
        # Record comprehensive metrics (same as API)
        metrics.record_request(
            success=success,
            response_time=response_time,
            model=model,
            k=int(k),
            error_type=error_type,
            avg_retrieval_distance=avg_distance,
            query_length=query_length,
            response_length=response_length,
            tokens_used=None
        )
        
        # Format distance for logging
        distance_str = f"{avg_distance:.4f}" if avg_distance is not None else "N/A"
        logger.info(
            f"UI Request completed in {response_time:.3f}s "
            f"(success={success}, avg_distance={distance_str})"
        )


def create_gradio_interface():
    """Create Gradio interface"""
    
    color = "orange"
    # Get custom CSS for the theme
    custom_css = get_custom_css(color)
    
    # Use Base theme with orange primary color
    theme = gr.themes.Base(
        primary_hue="orange",
        secondary_hue="gray",
        neutral_hue="slate",
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        slider_color="*primary_600",
    )
    
    with gr.Blocks(
        title="Disneyland Reviews Q&A (RAG)",
        theme=theme,
        css=custom_css
    ) as demo:
        
        gr.Markdown(
            """
            # 🎢 Disneyland Reviews Q&A (RAG)
            Ask questions and get answers grounded in actual visitor reviews from Hong Kong, Paris, and California Disneyland parks.
            """,
            elem_id="header"
        )
        
        query = gr.Textbox(
            label="Your question",
            placeholder="e.g., What do visitors like about Disneyland Hong Kong?",
            lines=3
        )
            
        with gr.Row():
            with gr.Column():
                top_k = gr.Slider(
                    minimum=5,
                    maximum=40,
                    value=20,
                    step=5,
                    label="Top-K passages",
                    info="Number of relevant reviews to retrieve"
                )
            with gr.Column():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    label="Creativity (temperature)",
                    info="Higher = more creative, Lower = more factual"
                )
        
        model = gr.Dropdown(
            choices=["gpt-4o-mini", "gpt-4o"],
            value="gpt-4o-mini",
            label="Model",
            info="GPT model to use"
        )
        
        ask_btn = gr.Button("Ask", variant="primary", size="lg")
        
        answer_out = gr.Textbox(
            label="Output",
            lines=20,
            interactive=False,
            show_copy_button=True
        )
        
        metadata_out = gr.Markdown(
            label="📊 Retrieval Info",
            visible=True
        )
        
        # Button click handler
        ask_btn.click(
            fn=query_rag,
            inputs=[query, top_k, temperature, model],
            outputs=[answer_out, metadata_out]
        )
        
        gr.Markdown(
            """
            ---
            **Built with Gradio** 🧡 • Use via [API](/docs) 🔗 • [Settings](#) ⚙️
            """
        )
    
    return demo

