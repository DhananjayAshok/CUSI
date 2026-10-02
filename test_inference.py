import requests
from PIL import Image
from io import BytesIO
from utils import (
    OpenAIModel,
    AnthropicModel,
    OpenRouterModel,
    HuggingFaceModel,
    OpenAITextEmbeddingModel,
    OpenRouterTextEmbeddingModel,
    HuggingFaceTextEmbeddingModel,
    HuggingFaceImageEmbeddingModel,
    JinaV4TextEmbeddingModel,
    JinaV4ImageEmbeddingModel,
    get_top_k_similars,
)


def get_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


image1_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTgIqqFyw989bp4cDVKGhEU8bJLeUJUIRIng&s"
image1 = get_img(image1_url)
image2 = get_img(
    "https://www.lifewire.com/thmb/FX4vAMlrOZIGACeuCjBKqgZeCKI=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/Screenshotfrom2019-01-1814-55-56-5c52207b46e0fb00014c3b34.png"
)

text1 = "The cat is on the table."
text2 = "Whats in this image?"

max_new_tokens = 20


def test_vlm(model):
    #print(model.infer(text1, images=[image1], max_new_tokens=max_new_tokens))
    #print(model.infer(text2, images=[image1, image2], max_new_tokens=max_new_tokens))
    print(
        model.infer(
            [text1, text2], images=[[image1], [image2]], max_new_tokens=max_new_tokens
        )
    )


def test_lm(model):
    #print(model.infer(text1, max_new_tokens=max_new_tokens))
    #print(model.infer([text2], max_new_tokens=max_new_tokens))
    print(model.infer([text1, text2], max_new_tokens=max_new_tokens))


def test_text_embedding(model):
    embedding1 = model.embed(texts=text1)
    embedding2 = model.embed(texts=text2)
    print("Embedding 1:", embedding1)
    print("Embedding 2:", embedding2)
    scores, _ = get_top_k_similars(
        query=embedding1, corpus=embedding2.unsqueeze(0), k=1
    )
    print("Similarity between text1 and text2:", scores[0].item())


def test_image_embedding(model):
    embedding1 = model.embed(images=image1)
    embedding2 = model.embed(images=image2)
    print("Embedding 1:", embedding1)
    print("Embedding 2:", embedding2)
    scores, _ = get_top_k_similars(
        query=embedding1, corpus=embedding2.unsqueeze(0), k=1
    )
    print("Similarity between image1 and image2:", scores[0].item())


models_to_test = {
    "lm": [
        HuggingFaceModel(model="meta-llama/Llama-3.2-1B-Instruct", model_kind="lm"),
        #OpenRouterModel(model="meta-llama/llama-3.3-70b-instruct"),
    ],
    "vlm": [
        #OpenAIModel(model="gpt-4o-mini"),  # PASSES
        #AnthropicModel(model="claude-haiku-4-5-20251001"),
        HuggingFaceModel(model="Qwen/Qwen3-VL-2B-Instruct", model_kind="vlm"),
    ],
    "text-embedding": [
        #OpenAITextEmbeddingModel(model="text-embedding-3-small"),
        #OpenRouterTextEmbeddingModel(model="nvidia/llama-nemotron-embed-vl-1b-v2:free"),
        HuggingFaceTextEmbeddingModel(model="Qwen/Qwen3-Embedding-0.6B"),
    ],
    "image-embedding": [
        HuggingFaceImageEmbeddingModel(model="Qwen/Qwen3-VL-Embedding-2B"),
    ],
}

for model_type, models in models_to_test.items():
    for model in models:
        print(f"\n--- Testing {model_type}: {model.model} ---")
        if model_type == "lm":
            test_lm(model)
        elif model_type == "vlm":
            test_lm(model)
            test_vlm(model)
        elif model_type == "text-embedding":
            test_text_embedding(model)
        elif model_type == "image-embedding":
            test_image_embedding(model)
