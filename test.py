import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN  # Import the trained model

# Load the trained model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("sensitive_classifier.pth"))
model.eval()

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class labels
classes = ['cats', 'tomatos']

# Function to test a single image
def test_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open the image
    transformed_image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    with torch.no_grad():
        outputs = model(transformed_image)
        _, predicted = torch.max(outputs, 1)
        class_label = classes[predicted.item()]

    print(f"Image: {image_path}, Predicted Class: {class_label}")
    return class_label

# Test multiple images
def test_multiple_images(test_folder):
    for filename in os.listdir(test_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Supported formats
            test_image(os.path.join(test_folder, filename))

if __name__ == "__main__":
    # Test a single image
    single_test_image = "dataset/test_image6.jpg"  # Replace with an actual image path
    test_image(single_test_image)

    # Test multiple images
    #test_folder = "dataset/test_samples"  # Replace with the path to your test images folder
    #test_multiple_images(test_folder)
