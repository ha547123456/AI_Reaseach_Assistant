"""
Sample Document Generator
Creates sample text documents for testing the AI Research Assistant
"""

import os

# Sample documents about AI and machine learning
sample_docs = {
    "doc1_intro_to_ai.txt": """
Artificial Intelligence: An Introduction

Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
especially computer systems. These processes include learning, reasoning, and self-correction. 
AI has become increasingly important in modern technology and is used in various applications 
from virtual assistants to autonomous vehicles.

The field of AI research was founded on the claim that human intelligence can be so precisely 
described that a machine can be made to simulate it. This raises philosophical arguments about 
the mind and the ethics of creating artificial beings with human-like intelligence.
""",
    
    "doc2_machine_learning.txt": """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on developing systems 
that can learn and improve from experience without being explicitly programmed. It uses 
algorithms to parse data, learn from it, and make determinations or predictions.

There are three main types of machine learning: supervised learning, unsupervised learning, 
and reinforcement learning. Each has its own use cases and benefits depending on the problem 
being solved.
""",
    
    "doc3_deep_learning.txt": """
Deep Learning and Neural Networks

Deep learning is a subset of machine learning that uses neural networks with multiple layers. 
These deep neural networks are capable of learning complex patterns in large amounts of data. 
The "deep" in deep learning refers to the number of layers in the neural network.

Applications of deep learning include image recognition, natural language processing, speech 
recognition, and many other domains. Deep learning has achieved remarkable success in tasks 
that were previously considered extremely difficult for computers.
""",
    
    "doc4_nlp.txt": """
Natural Language Processing

Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers 
understand, interpret, and manipulate human language. NLP draws from many disciplines, including 
computer science and computational linguistics, to bridge the gap between human communication 
and computer understanding.

Modern NLP applications include machine translation, sentiment analysis, chatbots, and text 
summarization. Recent advances in deep learning have significantly improved the capabilities 
of NLP systems.
""",
    
    "doc5_computer_vision.txt": """
Computer Vision Technology

Computer vision is a field of artificial intelligence that trains computers to interpret and 
understand the visual world. Using digital images from cameras and videos and deep learning 
models, machines can accurately identify and classify objects.

Applications include facial recognition, autonomous vehicles, medical image analysis, and 
augmented reality. Computer vision is one of the most rapidly advancing areas of AI research.
""",
    
    "doc6_reinforcement_learning.txt": """
Reinforcement Learning Principles

Reinforcement learning is a type of machine learning where an agent learns to make decisions 
by performing actions in an environment to maximize some notion of cumulative reward. Unlike 
supervised learning, the agent is not told which actions to take but must discover which 
actions yield the most reward through trial and error.

Famous applications include game playing (like AlphaGo), robotics, and autonomous systems. 
Reinforcement learning is particularly useful when the optimal solution is not known in advance.
""",
    
    "doc7_ai_ethics.txt": """
Ethics in Artificial Intelligence

As AI systems become more prevalent, ethical considerations become increasingly important. 
Issues include bias in AI algorithms, privacy concerns, job displacement, and the potential 
for misuse of AI technology. Ensuring AI systems are fair, transparent, and accountable is 
crucial for their responsible development and deployment.

Organizations and researchers are working on frameworks for ethical AI development, including 
principles of transparency, fairness, privacy protection, and human oversight. The goal is to 
create AI systems that benefit society while minimizing potential harms.
""",
    
    "doc8_ai_applications.txt": """
Real-World AI Applications

AI is transforming numerous industries and aspects of daily life. In healthcare, AI assists 
in diagnosis and treatment planning. In finance, it's used for fraud detection and algorithmic 
trading. In transportation, AI powers autonomous vehicles and traffic management systems.

Other applications include personalized recommendations in entertainment, smart home devices, 
customer service chatbots, and predictive maintenance in manufacturing. The breadth of AI 
applications continues to expand as the technology advances.
""",
    
    "doc9_neural_networks.txt": """
Understanding Neural Networks

Neural networks are computing systems inspired by biological neural networks in animal brains. 
They consist of interconnected nodes (neurons) organized in layers. Information flows through 
the network, with each neuron processing inputs and passing outputs to the next layer.

Different architectures exist for different tasks: Convolutional Neural Networks (CNNs) for 
image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for 
natural language understanding. Training neural networks involves adjusting connection weights 
to minimize error on training data.
""",
    
    "doc10_ai_future.txt": """
The Future of Artificial Intelligence

The future of AI holds tremendous potential and challenges. Researchers are working on 
achieving Artificial General Intelligence (AGI) - AI systems with human-level intelligence 
across all domains. While this remains a long-term goal, incremental advances continue to 
expand AI capabilities.

Emerging trends include edge AI (running AI on local devices), explainable AI (making AI 
decisions more transparent), and AI democratization (making AI tools accessible to non-experts). 
The integration of AI with other technologies like quantum computing and biotechnology may 
lead to breakthroughs we can't yet imagine.
""",
    
    "doc11_supervised_learning.txt": """
Supervised Learning Methods

Supervised learning is a machine learning paradigm where the algorithm learns from labeled 
training data. The algorithm receives input-output pairs and learns to map inputs to correct 
outputs. Common algorithms include linear regression, logistic regression, decision trees, 
random forests, and support vector machines.

Applications include spam detection, image classification, price prediction, and medical 
diagnosis. The quality of supervised learning models heavily depends on having sufficient 
high-quality labeled training data.
""",
    
    "doc12_unsupervised_learning.txt": """
Unsupervised Learning Techniques

Unsupervised learning works with unlabeled data to discover hidden patterns or structures. 
Unlike supervised learning, there are no predefined correct answers. Common techniques include 
clustering (grouping similar items), dimensionality reduction, and anomaly detection.

Applications include customer segmentation, recommendation systems, and data compression. 
Unsupervised learning is particularly valuable when labeled data is expensive or impossible 
to obtain.
""",
    
    "doc13_ai_algorithms.txt": """
Key AI Algorithms

Various algorithms power different AI applications. Classification algorithms categorize data 
into predefined classes. Regression algorithms predict continuous values. Clustering algorithms 
group similar data points. Decision trees make decisions through tree-like models.

Ensemble methods combine multiple algorithms for better performance. Gradient boosting, random 
forests, and neural networks are popular choices. The selection of algorithm depends on the 
problem type, data characteristics, and performance requirements.
""",
    
    "doc14_data_science.txt": """
Data Science and AI

Data science is the foundation of modern AI systems. It involves extracting insights from 
structured and unstructured data using scientific methods, algorithms, and systems. Data 
scientists prepare and analyze data to train AI models and evaluate their performance.

The data science pipeline includes data collection, cleaning, exploration, feature engineering, 
model training, evaluation, and deployment. Good data science practices are essential for 
building reliable and effective AI systems.
""",
    
    "doc15_ai_challenges.txt": """
Current Challenges in AI

Despite remarkable progress, AI faces several challenges. Data quality and availability remain 
significant issues. Many AI systems require massive amounts of labeled data, which is expensive 
to create. AI models can be opaque, making it difficult to understand their decision-making 
process.

Other challenges include adversarial attacks, bias and fairness concerns, energy consumption 
of large models, and the need for specialized hardware. Addressing these challenges is crucial 
for advancing AI technology responsibly and sustainably.
"""
}

def create_sample_documents(output_dir="sample_documents"):
    """Create sample documents in the specified directory"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Write each document
    for filename, content in sample_docs.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"Created: {filepath}")
    
    print(f"\n Successfully created {len(sample_docs)} sample documents in '{output_dir}' directory")
    print(f" You can now upload these files to the AI Research Assistant application")

if __name__ == "__main__":
    create_sample_documents()
