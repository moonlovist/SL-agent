import base64
from langchain_core.messages import HumanMessage, SystemMessage


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_message(system_prompt, user_prompt, image_path=None):
    system_message = SystemMessage(content=system_prompt)

    content = [{"type": "text", "text": user_prompt}]

    if image_path:
        image_paths = [image_path] if isinstance(image_path, str) else image_path

        for img_path in image_paths:
            base64_image = image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

    human_message = HumanMessage(content=content)
    return [system_message, human_message]
