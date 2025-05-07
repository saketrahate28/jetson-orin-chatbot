const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');
const fetch = require('node-fetch');

// Load environment variables from .env file
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// NIM API Configuration
const NIM_CONFIG = {
    API_KEY: process.env.NIM_API_KEY,
    BASE_URL: process.env.NIM_API_URL || 'https://api.nim.nvidia.com',
    MODEL: process.env.NIM_MODEL || 'llama3-70b-instruct',
    MAX_TOKENS: parseInt(process.env.NIM_MAX_TOKENS || '2048'),
    TEMPERATURE: parseFloat(process.env.NIM_TEMPERATURE || '0.7')
};

// Check if NIM API key is configured
if (!NIM_CONFIG.API_KEY) {
    console.error("ERROR: NIM_API_KEY is not set in .env file");
    console.log("Please add your NVIDIA NIM API key to the .env file");
    console.log("Example: NIM_API_KEY=your_api_key_here");
}

// Status endpoint
app.get('/api/status', async (req, res) => {
    console.log("Status check requested");

    if (!NIM_CONFIG.API_KEY) {
        return res.status(500).json({
            status: 'error',
            message: 'API key not configured in .env file'
        });
    }

    try {
        // Test connection to NIM API
        const testResult = await testNimConnection();

        if (testResult.success) {
            return res.json({
                status: 'connected',
                model: NIM_CONFIG.MODEL
            });
        } else {
            return res.status(401).json({
                status: 'error',
                message: testResult.error || 'Failed to connect to NIM API'
            });
        }
    } catch (error) {
        console.error("Error checking NIM status:", error);
        return res.status(500).json({
            status: 'error',
            message: error.message
        });
    }
});

// Chat endpoint
app.post('/api/chat', async (req, res) => {
    console.log("Chat request received");
    const { messages } = req.body;

    if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({
            status: 'error',
            message: 'Invalid request: messages array is required'
        });
    }

    if (!NIM_CONFIG.API_KEY) {
        return res.status(500).json({
            status: 'error',
            message: 'API key not configured in .env file'
        });
    }

    try {
        const response = await fetch(`${NIM_CONFIG.BASE_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
            },
            body: JSON.stringify({
                model: NIM_CONFIG.MODEL,
                messages: messages,
                max_tokens: NIM_CONFIG.MAX_TOKENS,
                temperature: NIM_CONFIG.TEMPERATURE
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error("NIM API error:", errorData);
            return res.status(response.status).json({
                status: 'error',
                message: errorData.error?.message || 'Error from NIM API'
            });
        }

        const data = await response.json();

        return res.json({
            status: 'success',
            message: data.choices[0]?.message?.content || ''
        });
    } catch (error) {
        console.error("Error calling NIM API:", error);
        return res.status(500).json({
            status: 'error',
            message: error.message
        });
    }
});

// Test NIM connection
async function testNimConnection() {
    try {
        // Minimal request to test connection
        const response = await fetch(`${NIM_CONFIG.BASE_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
            },
            body: JSON.stringify({
                model: NIM_CONFIG.MODEL,
                messages: [
                    { role: "system", content: "You are a helpful assistant." },
                    { role: "user", content: "Hello, are you connected?" }
                ],
                max_tokens: 5
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            return {
                success: false,
                error: errorData.error?.message || `API error: ${response.status}`
            };
        }

        const data = await response.json();
        return {
            success: true,
            data: data
        };
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

// Start server
app.listen(PORT, () => {
    console.log(`
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   L4T Chatbot Backend Server                     │
    │                                                  │
    │   Server running on http://localhost:${PORT}        │
    │                                                  │
    │   NIM API Key: ${NIM_CONFIG.API_KEY ? '✓ Configured' : '✗ Not Configured'} │
    │   NIM Model: ${NIM_CONFIG.MODEL}           │
    │                                                  │
    └──────────────────────────────────────────────────┘
    `);
});