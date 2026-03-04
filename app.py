"""
ModernBERT-RGAT | Gradio Web Demo
====================================
Interactive web app for Aspect-Based Sentiment Analysis.

Run with:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def create_app():
    """Build and return the Gradio app interface."""
    import gradio as gr
    from src.inference import load_predictor

    # --- Load model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = None

    for year in ['2014', '2015', '2016']:
        ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', f'best_model_{year}.pt')
        if os.path.exists(ckpt):
            print(f"Loading model from: {ckpt}")
            predictor = load_predictor(year=year, device=device)
            break

    if predictor is None:
        raise FileNotFoundError(
            "No checkpoints found. Train a model first using notebooks/04_training.ipynb"
        )

    # --- Prediction function ---
    def analyze_review(text: str) -> tuple:
        """Run inference and return highlighted HTML + structured results."""
        if not text or not text.strip():
            return "<p style='color: #999;'>Enter a restaurant review above.</p>", ""

        predictions = predictor.predict(text.strip())

        if not predictions:
            html = (
                f'<div style="font-size: 16px; padding: 16px; background: #f8f9fa; '
                f'border-radius: 8px; line-height: 2;">{text}</div>'
                f'<p style="color: #999; margin-top: 8px;">No aspects detected.</p>'
            )
            return html, "No aspects detected in this text."

        # Highlighted HTML
        highlighted = predictor.get_highlighted_html(text.strip(), predictions)
        html = (
            f'<div style="font-size: 16px; padding: 16px; background: #f8f9fa; '
            f'border-radius: 8px; line-height: 2.2;">{highlighted}</div>'
        )

        # Legend
        html += '''
        <div style="margin-top: 12px; font-size: 13px;">
          <b>Legend:</b>
          <span style="background: #27ae60; color: white; padding: 2px 8px; border-radius: 4px;">Positive</span>
          <span style="background: #e74c3c; color: white; padding: 2px 8px; border-radius: 4px;">Negative</span>
          <span style="background: #f39c12; color: white; padding: 2px 8px; border-radius: 4px;">Neutral</span>
          <span style="background: #8e44ad; color: white; padding: 2px 8px; border-radius: 4px;">Conflict</span>
        </div>
        '''

        # Text summary
        lines = [f"Found {len(predictions)} aspect(s):\n"]
        emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐', 'conflict': '🤔'}
        for p in predictions:
            e = emoji.get(p.sentiment, '❓')
            lines.append(f"  {e} \"{p.aspect}\" → {p.sentiment} (confidence: {p.confidence:.2f})")

        return html, "\n".join(lines)

    # --- Build Gradio UI ---
    examples = [
        "The spicy ramen was incredibly flavorful and the broth was rich.",
        "Terrible pizza with a soggy crust, but the drinks were excellent.",
        "Average food, nothing special about the ambiance either.",
        "The sushi here is the best I have ever had, fresh and perfectly seasoned.",
        "Long wait times and rude staff ruined an otherwise decent meal.",
        "Loved the cheesecake but the coffee was lukewarm and bitter.",
        "Great location with a beautiful patio, although prices are a bit high.",
        "The butter chicken was creamy and aromatic, paired perfectly with garlic naan.",
    ]

    with gr.Blocks(
        title="ModernBERT-RGAT | Aspect Sentiment Analysis",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .main-header h1 { color: #2c3e50; }
        .main-header p { color: #7f8c8d; font-size: 16px; }
        """
    ) as app:
        gr.HTML("""
        <div class="main-header">
            <h1>🍽️ ModernBERT-RGAT</h1>
            <p>Joint Aspect Extraction & Sentiment Classification for Restaurant Reviews</p>
            <p style="font-size: 13px; color: #95a5a6;">
                Powered by ModernBERT + Relational Graph Attention Networks
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Restaurant Review",
                    placeholder="Type a restaurant review here...",
                    lines=3,
                    max_lines=6,
                )
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                gr.Examples(
                    examples=[[ex] for ex in examples],
                    inputs=text_input,
                    label="Example Reviews",
                )

            with gr.Column(scale=1):
                html_output = gr.HTML(label="Highlighted Analysis")
                text_output = gr.Textbox(label="Prediction Details", lines=5, interactive=False)

        analyze_btn.click(
            fn=analyze_review,
            inputs=text_input,
            outputs=[html_output, text_output],
        )
        text_input.submit(
            fn=analyze_review,
            inputs=text_input,
            outputs=[html_output, text_output],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
