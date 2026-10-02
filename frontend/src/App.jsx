import { useState } from "react";
import "./App.css";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setProbability(null);
    setError("");
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setPreview(null);
    setPrediction(null);
    setProbability(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError("Please select a spiral image first.");
      return;
    }

    setLoading(true);
    setPrediction(null);
    setProbability(null);
    setError("");

    const formData = new FormData();

    // Uploaded image
    formData.append("image", selectedImage);

    // Tell backend that spiral-image validation should be performed
    formData.append("validate_spiral", "true");

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/api/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Prediction failed.");
      }

      setPrediction(data.prediction);
      setProbability(data.probability);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Refresh the page for a new prediction
  const handleRefresh = () => {
    window.location.reload();
  };

  const probabilityPercentage =
    probability !== null
      ? (probability * 100).toFixed(2)
      : "0.00";

  return (
    <div className="app">

      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar-inner">

          <div className="logo-area">
            <div className="logo-mark">N</div>

            <div className="logo-text">
              <div className="logo-title">NeuroScan</div>
              <div className="logo-subtitle">
                Parkinson's Screening
              </div>
            </div>
          </div>

          <div className="online-status">
            <span className="online-dot"></span>
            System Online
          </div>

        </div>
      </nav>


      {/* Main */}
      <main className="main-content">

        {/* Hero */}
        <section className="hero-section">

          <div className="hero-badge">
            PARKINSON'S HEALTH SCREENING
          </div>

          <h1>
            Parkinson's Screening
            <span>Through Spiral Drawings</span>
          </h1>

          <p className="hero-description">
            Upload a spiral drawing and receive a screening
            result based on the image.
          </p>

        </section>


        {/* Upload Card */}
        <section className="screening-card">

          <div className="section-heading">
            <div className="heading-icon">
              ↑
            </div>

            <div>
              <h2>Upload Spiral Image</h2>

              <p>
                Select a clear spiral drawing to begin screening.
              </p>
            </div>
          </div>


          {!preview && (
            <label className="upload-box">

              <div className="upload-symbol">
                ↑
              </div>

              <h3>Select your spiral image</h3>

              <p>
                Drag and drop your image here or choose a file
              </p>

              <span className="browse-button">
                Choose Image
              </span>

              <small>
                PNG, JPG or JPEG
              </small>

              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleImageChange}
              />

            </label>
          )}


          {preview && (
            <div className="selected-image-area">

              <div className="selected-image-header">

                <div>
                  <h3>Selected Spiral Image</h3>

                  <p>
                    {selectedImage.name}
                  </p>
                </div>

                <button
                  className="remove-button"
                  onClick={handleRemoveImage}
                >
                  Remove
                </button>

              </div>


              <div className="image-container">

                <img
                  src={preview}
                  alt="Selected spiral drawing"
                />

              </div>


              <button
                className="analyze-button"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze Spiral"}
              </button>

            </div>
          )}


          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

        </section>


        {/* Result */}
        {prediction && (
          <section className="result-card">

            <div className="result-title">
              <span>SCREENING RESULT</span>
              <h2>Your Result</h2>
            </div>


            <div
              className={
                prediction === "Parkinson's"
                  ? "result-box possible"
                  : "result-box healthy"
              }
            >

              <div className="result-symbol">
                {prediction === "Parkinson's" ? "!" : "✓"}
              </div>

              <div className="result-text">
                <p>Screening Classification</p>
                <h3>{prediction}</h3>
              </div>

            </div>


            <div className="score-area">

              <div className="score-header">
                <span>Screening Score</span>

                <strong>
                  {probabilityPercentage}%
                </strong>
              </div>

              <div className="score-track">
                <div
                  className="score-fill"
                  style={{
                    width: probabilityPercentage + "%",
                  }}
                ></div>
              </div>

            </div>


            <div className="hope-card">

              <div className="heart-icon">♥</div>

              <div>
                <h3>A Message of Hope</h3>

                <p>
                  Every step forward is a victory. Keep going,
                  keep believing, and never let Parkinson's
                  define the strength within you.
                </p>
              </div>

            </div>


            {/* Refresh Button */}
            <button
              className="refresh-button"
              onClick={handleRefresh}
            >
              ↻ Analyze Another Image
            </button>

          </section>
        )}

      </main>


      {/* Footer */}
      <footer className="footer">

        <div className="footer-inner">

          <div className="footer-brand">
            <strong>NeuroScan</strong>
            <span>Parkinson's Screening</span>
          </div>

          <p>
            Supporting awareness through technology.
          </p>

        </div>

      </footer>

    </div>
  );
}

export default App;