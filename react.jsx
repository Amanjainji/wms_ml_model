import React, { useState } from 'react';

const PredictBinStatus = () => {
    const [features, setFeatures] = useState({
        "Gas Sensor (ppm)": "",
        "Temperature Sensor (°C)": "",
        "Light Sensor (lux)": "",
        "Ultrasonic Sensor (cm)": "",
        "Moisture Sensor (%)": "",
        "Weight Sensor (kg)": ""
    });
    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        setFeatures({ ...features, [e.target.name]: e.target.value });
    };

    const handlePredict = async () => {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: Object.values(features).map(Number) })
        });
        const data = await response.json();
        setPrediction(data);
    };

    return (
        <div className="p-5">
            <h2 className="text-xl font-bold">Smart Bin Prediction</h2>
            <div className="grid grid-cols-2 gap-4 mt-4">
                {Object.keys(features).map((key) => (
                    <div key={key}>
                        <label className="block text-sm font-medium">{key}</label>
                        <input
                            type="number"
                            name={key}
                            value={features[key]}
                            onChange={handleChange}
                            className="border p-2 w-full"
                        />
                    </div>
                ))}
            </div>
            <button onClick={handlePredict} className="mt-4 bg-blue-500 text-white px-4 py-2 rounded">
                Predict
            </button>

            {prediction && (
                <div className="mt-5 p-4 bg-gray-100 rounded">
                    <h3 className="text-lg font-bold">Prediction Results:</h3>
                    <p><strong>Bin Status:</strong> {prediction["Bin Status"] === 1 ? 'Needs Collection' : 'OK'}</p>
                    <p><strong>Percentage of Full:</strong> {prediction["Percentage of Full (%)"]}%</p>
                    <p><strong>Bins Requiring Attention:</strong> {prediction["Bins Requiring Attention (%)"]}%</p>
                    <p><strong>Collection Efficiency:</strong> {prediction["Collection Efficiency (%)"]}%</p>
                    <p><strong>Total Waste Collected:</strong> {prediction["Total Waste Collected (kg)"]} kg</p>
                    <p><strong>Average Collection Rate per Day:</strong> {prediction["Average Collection Rate per Day"]} kg/day</p>
                    <p><strong>Recycling Rate:</strong> {prediction["Recycling Rate (%)"]}%</p>
                </div>
            )}
        </div>
    );
};

export default PredictBinStatus;
