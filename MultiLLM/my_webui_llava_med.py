import gradio as gr 
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

model_id = "/mnt/newdisk/KJY/MultiLLM/models/llava1_5-med-7b-en/"
# model_id = "/mnt/newdisk/MultiLLM_models/llava-hf-7b/llava-1.5-7b-hf/"
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

def predict(history, image, question):
    # Process the image and question
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    # Move inputs to the GPU
    inputs = inputs.to("cuda")

    # Generate output using the model
    output_ids = model.generate(**inputs,
                                max_new_tokens=1024,  # 增加生成文本的最大长度
                                do_sample=True,  # 启用采样
                                temperature=0.7,  # 设置采样温度，值越高内容越随机
                                top_p=0.9,  # Nucleus sampling，控制生成文本质量
                                top_k=50,  # 限制采样范围，较高值适合生成长文本
                                use_cache=True
                                )

    # Extract generated tokens (remove input part)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    # Decode generated tokens to readable text
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    # Append the current question and output to history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": output_text})

    return history

# Create Gradio interface
with gr.Blocks(css=".gradio-container {background-color: lightblue}") as demo:
    gr.Markdown("<h2 style='text-align: center;'>llava1.5-med-7B</h2>")
    
    # Add an elem_id to the Chatbot component for custom styling
    chat_history_output = gr.Chatbot(label="Conversation Record", type='messages', height=800, elem_id="chat_history") 
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image", elem_id="image_input")  # English label and elem_id
        question_input = gr.Textbox(placeholder="Ask something about the image...", label="Your Question")  # English placeholder
    
    submit_btn = gr.Button("Submit")
    submit_btn.click(predict, inputs=[chat_history_output, image_input, question_input], outputs=[chat_history_output])

    # Add custom CSS
    gr.HTML("""
    <style>
        #image_input {
            width: 300px;  /* Fixed width */
            height: 300px; /* Fixed height */
            border: 2px dashed #cccccc; /* Border style */
        }
    </style>
    """)

# Launch the Gradio app
demo.launch(debug=True)
