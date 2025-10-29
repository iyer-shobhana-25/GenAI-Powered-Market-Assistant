# 🤖 GenAI-Powered Market Assistant

A Finance-Specific GenAI Assistant that:
• Uses the trained model to forecast VIX,
• Interprets market movements,
• Answers financial queries related to NIFTY, VIX, and volatility,
• Uses relevant finance terminology.

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

Clone the project files to your local machine:

```bash
git clone https://github.com/iyer-shobhana-25/GenAI-Powered-Market-Assistant
cd GenAI-Powered-Market-Assistant
````

*(Replace `https://github.com/iyer-shobhana-25/GenAI-Powered-Market-Assistant` and `GenAI-Powered-Market-Assistant` with your project's actual details.)*

### 2\. Set Up Environment Variables

The application requires two environment variables to function. You must set these **before** running the code.

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| **`GOOGLE_API_KEY`** | Your API key for accessing Google's models. | `AIzaSy...` |
| **`MODEL_NAME`** | The specific model to use for the chatbot (e.g., Gemini). | `gemini-2.5-flash` |

**Example for Linux/macOS:**

```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
export MODEL_NAME="gemini-2.5-flash"
```

**Example for Windows (Command Prompt):**

```bash
set GOOGLE_API_KEY="YOUR_API_KEY_HERE"
set MODEL_NAME="gemini-2.5-flash"
```

Or Create an .env file for Same

### 3\. Install Dependencies

Install the required Python packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4\. Run the Application

Start the web server using the main Python file:

```bash
python finance_genai_assistant.py
```


-----

## 🌐 Access

Once the application is running, you can access the following in your web browser:

  * **Chatbot Interface:** You can interact with the live chatbot at:
    **`http://localhost:1525`**

  * **API Documentation:** The full API documentation, including all available endpoints, is accessible at:
    **`http://localhost:1525/docs`**

-----

## 🛠 Project Structure

  * `finance_genai_assistant.py`: The main server application file.
  * `requirements.txt`: A list of all project dependencies.

-----

## 🤝 Contribution

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## 📄 License

This project is licensed under the [LICENSE NAME] License - see the `LICENSE.md` file for details.

```

Would you like to customize any section of this README, or add more specific details about your project's features?
```