from transformers import pipeline
import gradio as gr

pipe = pipeline("toxic-comment", model="jasurbek-fm/toxic-comment-distelbert")

demo = gr.Interface.from_pipeline(pipe)
demo.launch()
