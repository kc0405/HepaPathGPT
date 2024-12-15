import torch
from transformers import AutoModelForCausalLM
import gradio as gr 
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


# specify the path to the model
model_path = "/mnt/newdisk/KJY/swift/output/deepseek-vl-7b-chat/en/deepseek-vl-7b-chat/v0-20241114-155749/checkpoint-12723-merged/"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def predict(history, image, question):
    # Process the image and question
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{question}",
            "images": []
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[image],
        force_batchify=True
    ).to(vl_gpt.device)

    # Run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,  # 增加生成文本的最大长度
        do_sample=True,  # 启用采样
        temperature=0.7,  # 设置采样温度，值越高内容越随机
        top_p=0.9,  # Nucleus sampling，控制生成文本质量
        top_k=50,  # 限制采样范围，较高值适合生成长文本
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    # Append the current question and output to history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    return history

# Create Gradio interface
with gr.Blocks(css=".gradio-container {background-color: lightblue}") as demo:
    gr.Markdown("<h2 style='text-align: center;'>deepseek-vl-7b-chat</h2>")
    
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
