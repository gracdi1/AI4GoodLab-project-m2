// app/page.tsx
'use client'
import React, { useState } from 'react'
import Image from 'next/image';

export default function Home() {
  const [prostheticType, setProstheticType] = useState("");
  const [legSide, setLegSide] = useState('');
  const [exerciseList, setExerciseList] = useState('');
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState('No file chosen');
  const [exercise, setExercise] = useState('N/A');
  const [prostheticOutput, setProstheticOutput] = useState('N/A');
  const [purpose, setPurpose] = useState('N/A');
  const [mistakes, setMistakes] = useState('N/A');
  const [exerciseSteps, setExerciseSteps] = useState<string[]>([]);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackContent, setFeedbackContent] = useState('');
  const [showWebcam, setShowWebcam] = useState(false);
  const [currentStep, setCurrentStep] = useState('N/A');
  const [vlmFeedback, setVlmFeedback] = useState('Awaiting feedback...');
  const [vlmStatus, setVlmStatus] = useState('Idle');
  const [videoUploaded, setVideoUploaded] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  //const [selectedTab, setSelectedTab] = useState<'exercise' | 'recommend'>('exercise');


  const handleAskClick = async () => {
  const formData = new FormData();
  formData.append('prosthetic_type', prostheticType);
  formData.append('leg', legSide);
  formData.append('exercises', exerciseList);
  if (pdfFile) {
    formData.append('pdf_file', pdfFile);
  }

  try {
    const res = await fetch('/ask_llm', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    console.log('Received:', data);

    setExercise(data.exercise || 'N/A');
    setProstheticOutput(data.prosthetic || 'N/A');
    setPurpose(data.purpose || 'N/A');
    setMistakes(data.mistakes || 'N/A');
    setExerciseSteps(data.steps || []); // assuming steps is an array of strings

  } catch (err) {
    console.error('Error:', err);
  }
};


  return (
    <main>
        <Image
        src="/images/VEHAB-Logo.png"
        alt="VEHAB Logo"
        width={400}
        height={200}
        priority
      />
      <h1>VEHAB: Your AI Prosthetic Rehab Assistant</h1>
      {/*  TAB CONTROLS  */}
      <div className="tabs">
        <button className="tab-button active" data-target="exerciseTab">
          Exercise
        </button>
        <button className="tab-button" data-target="recommendTab">
          Get Recommendations
        </button>
      </div>
      {/*  EXERCISE TAB  */}
      <section id="exerciseTab" className="tab-content">
        <h2>Step-by-Step Exercises</h2>
        <p>
          Describe your prosthetic/amputation type to get tailored exercise
          recommendations:
        </p>
        <select
          value={prostheticType} onChange={(e) => setProstheticType(e.target.value)}>
          <option value="" disabled>
            Select prosthetic/amputation type
          </option>
          <option value="transtibial">Transtibial (below knee)</option>
          <option value="transfemoral">Transfemoral (above knee)</option>
        </select>

        <select
          value={legSide}
          onChange={(e) => setLegSide(e.target.value)}
        >
          <option value="" disabled>
            Select right or left leg
          </option>
          <option value="right">Right leg</option>
          <option value="left">Left leg</option>
        </select>
        <br />

        <p>List the exercises recommended to you by your physiotherapist:</p>
        <textarea
          value={exerciseList}
          onChange={(e) => setExerciseList(e.target.value)}
          placeholder="e.g., Hamstring Stretch, Hip Extension on Side, Side Stepping."
        />
        <br />

        <div className="button-row">
          <button onClick={handleAskClick}>Get Recommendation</button>
          <br />

          <div className="file-upload-inline-container">
            <input
              id="inlinePdfFileInput"
              type="file"
              name="pdf_file"
              accept=".pdf"
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                setPdfFile(file);
                setFileName(file?.name || 'No file chosen');
              }}
            />
            <label htmlFor="inlinePdfFileInput" className="custom-file-upload">
              Upload PDF
            </label>
            <span className="file-name">{fileName}</span>
          </div>

          <button onClick={() => {
            setPdfFile(null);
            setFileName('No file chosen');
          }}>
            Cancel Upload
          </button>
        </div>

        <p style={{ marginTop: "0.5rem", fontWeight: "bold" }}>{uploadStatus}</p>

        <div style={{ marginTop: "1rem" }}>
          <p>
            <strong>Exercise:</strong> <span>{exercise}</span>
          </p>
          <p>
            <strong>Prosthetic limb type(s):</strong> <span>{prostheticOutput}</span>
          </p>
          <p>
            <strong>Exercise Purpose:</strong> <span>{purpose}</span>
          </p>
          <p>
            <strong>Common Mistakes:</strong> <span>{mistakes}</span>
          </p>

          <div className="exercise-steps-container">
            <h4>Exercise Steps:</h4>
            <ul>
              {exerciseSteps.length > 0 ? (
                exerciseSteps.map((step, idx) => <li key={idx}>{step}</li>)
              ) : (
                <li>Steps will appear here.</li>
              )}
            </ul>

            <div className="button-group">
              <button className="hidden" onClick={() => setShowWebcam(true)}>
                Start Exercise with VLM
              </button>

              <button className="hidden" onClick={() => setVideoUploaded(true)}>
                Upload Video
              </button>

              <input
                type="file"
                accept="video/*"
                style={{ display: "none" }}
                onChange={(e) => {
                  // Placeholder logic
                  console.log("Video file selected:", e.target.files?.[0]);
                }}
              />

              <button className="hidden" onClick={() => setShowFeedback(true)}>
                Analyze Video
              </button>

              <button className="hidden secondary" onClick={() => setCurrentStep('Next step placeholder')}>
                Next Step
              </button>

              <button className="hidden secondary" onClick={() => alert("Exercise complete!")}>
                Finish Exercise
              </button>
            </div>


            <div className="video-preview" />
          </div>

          <div className="error" style={{ display: "none" }} />
        </div>



          {/* Feedback section - only shows AFTER successful video analysis */}
          {/* Feedback section - only shows AFTER successful video analysis */}
          {showFeedback && (
            <div className="feedback-section show">
              <h3>Exercise Analysis Results</h3>
              <div className="feedback-content">{feedbackContent}</div>
            </div>
          )}

          {/* Webcam Section */}
          {showWebcam && (
            <div className="webcam-container">
              <h3>Exercise Supervision (Webcam)</h3>
              <video id="videoElement" autoPlay muted playsInline />
              <div className="vlm-feedback">
                <h4>
                  Current Step: <span>{currentStep}</span>
                </h4>
                <p>{vlmFeedback}</p>
                <p className="vlm-status">
                  Status: <span>{vlmStatus}</span>
                </p>
              </div>
              <button
                className="secondary"
                onClick={() => {
                  // stop webcam logic
                  setShowWebcam(false);
                }}
              >
                Stop Webcam
              </button>
            </div>
          )}

          {/* Video Analysis Container */}
          {videoUploaded && (
            <div id="video-analysis-container">
              <div className="video-display">
                <h3>Uploaded Video</h3>
                <video id="analysis-video" controls />
              </div>
              <div className="analysis-feedback">
                <h3>Analysis Results</h3>
                <div id="analysis-content">
                  <span className="analyzing-spinner">⟳</span> Analyzing video...
                </div>
              </div>
            </div>
          )}

                </section>

      {/* SECOND TAB - placeholder for now */}
      <section id="recommendTab" className="tab-content hidden">
        <h2>Get Recommendations</h2>
        <p>Get basic exercises based on recovery timeline, skill, and prosthetic type</p>
      </section>
    </main>
  );
}

