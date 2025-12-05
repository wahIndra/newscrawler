# Indonesian News Sentiment Analyzer v3.0

A powerful, AI-powered Streamlit application that crawls news, analyzes sentiment using advanced ensemble models, and provides analyst-style insights with trend forecasting.

![App Screenshot](https://github.com/wahIndra/newscrawler/blob/main/assets/app_screenshot.png?raw=true)

## Key Features (v3.0)

*   **Multi-Language Support**: Search and analyze news in **Bahasa Indonesia**, **English**, or **Both** simultaneously.
*   **Advanced Sentiment Ensemble**:
    *   Combines **IndoBERT** (ID), **XLM-RoBERTa** (Multilingual), and **DistilBERT** (EN).
    *   **Ensemble Mode**: Averages predictions from selected models for higher accuracy.
*   **Analyst-Style Insights**:
    *   **Automated Explanations**: Generates "Analyst Notes" explaining *why* a sentiment was chosen.
    *   **Sentiment Markers**: Highlights positive/negative keywords (including **slang** like *cuan, mantul*) in the text.
    *   **Massive Keyword Library**: Over **2000+ keywords** for robust detection across formal and informal text.
*   **Trend Forecasting**:
    *   **3-Month Forecast**: Projects sentiment trends 90 days into the future using linear regression.
*   **Smart Search**:
    *   **Date Range & Article Count**: Flexible search options (up to 100 articles).
    *   **Robust Scraping**: Handles Google News redirects and anti-scraping measures.
*   **Premium UI**: Modern, dark-themed interface with interactive charts and expandable details.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/wahIndra/newscrawler.git
    cd newscrawler
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser** at `http://localhost:8501`.

3.  **Configure Search**:
    *   **Topic**: Enter a keyword (e.g., "Ekonomi Indonesia", "Timnas").
    *   **Language**: Select ID, EN, or Both.
    *   **Models**: Choose one or multiple models (Ensemble).
    *   **Settings**: Adjust date range and article limit.

4.  **Analyze**: Click "Analyze Sentiment" to see results, trends, and forecasts.

## Project Structure

*   `app.py`: Main Streamlit application.
*   `crawler/`:
    *   `searcher.py`: Google News search logic.
    *   `scraper.py`: Article content extraction.
*   `model/`:
    *   `sentiment.py`: Sentiment analysis logic (Ensemble, Explanations).
    *   `keywords.py`: Extensive keyword library (2000+ words).
*   `verify_components.py`: Verification script for testing features.

## License

MIT License - see [LICENSE](LICENSE) for details.
