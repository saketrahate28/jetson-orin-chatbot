/**
 * NVIDIA Inference Microservices (NIM) API Client
 *
 * This file provides functions for interacting with NVIDIA NIM services.
 * It depends on nim-config.js being loaded first.
 */

// Check if NIM_CONFIG is defined
if (typeof NIM_CONFIG === 'undefined') {
    console.error("NIM_CONFIG is not defined. Make sure nim-config.js is loaded before nim-api.js");
    throw new Error("NIM_CONFIG is not defined");
}

/**
 * NIM API Client
 */
const NimAPI = {
    /**
     * Send a chat completion request to NIM
     * @param {Array} messages - Array of message objects with role and content
     * @param {Object} options - Optional parameters for the request
     * @returns {Promise} - Promise that resolves to the response from NIM
     */
    async chatCompletion(messages, options = {}) {
        if (!Array.isArray(messages) || messages.length === 0) {
            throw new Error("Messages must be a non-empty array");
        }

        const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                'Content-Type': 'application/json'
                    },
            body: JSON.stringify({
                messages: messages,
                model: options.model || NIM_CONFIG.MODELS.CHAT,
                temperature: options.temperature || NIM_CONFIG.DEFAULTS.TEMPERATURE,
                max_tokens: options.max_tokens || NIM_CONFIG.DEFAULTS.MAX_TOKENS,
                stream: options.stream !== undefined ? options.stream : NIM_CONFIG.DEFAULTS.STREAM
            })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error?.message || `API error: ${response.status}`);
                }

                return await response.json();
    },

    /**
     * Generate embeddings for a text or array of texts
     * @param {string|Array} input - Text or array of texts to generate embeddings for
     * @param {Object} options - Optional parameters for the request
     * @returns {Promise} - Promise that resolves to the embeddings
     */
    async createEmbeddings(input, options = {}) {
        const texts = Array.isArray(input) ? input : [input];

        const params = {
            model: options.model || NIM_CONFIG.MODELS.EMBEDDINGS,
            input: texts
        };

        try {
            const response = await fetch(`${NIM_CONFIG.ENDPOINTS.BASE_URL}${NIM_CONFIG.ENDPOINTS.EMBEDDINGS}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
                },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error?.message || `API error: ${response.status}`);
            }

            const data = await response.json();
            return Array.isArray(input) ? data.data : data.data[0];
        } catch (error) {
            console.error("NIM embeddings error:", error);
            throw error;
        }
    },

    /**
     * Send a text completion request to NIM
     * @param {string} prompt - The prompt to generate completion for
     * @param {Object} options - Optional parameters for the request
     * @returns {Promise} - Promise that resolves to the response from NIM
     */
    async textCompletion(prompt, options = {}) {
        if (!prompt || typeof prompt !== 'string') {
            throw new Error("Prompt must be a non-empty string");
        }

        const params = {
            model: options.model || NIM_CONFIG.MODELS.TEXT,
            prompt: prompt,
            max_tokens: options.max_tokens || NIM_CONFIG.DEFAULTS.MAX_TOKENS,
            temperature: options.temperature || NIM_CONFIG.DEFAULTS.TEMPERATURE,
            top_p: options.top_p || NIM_CONFIG.DEFAULTS.TOP_P,
            frequency_penalty: options.frequency_penalty || NIM_CONFIG.DEFAULTS.FREQUENCY_PENALTY,
            presence_penalty: options.presence_penalty || NIM_CONFIG.DEFAULTS.PRESENCE_PENALTY,
            stream: options.stream !== undefined ? options.stream : NIM_CONFIG.DEFAULTS.STREAM
        };

        // Handle streaming similar to chatCompletion if needed
        if (params.stream && options.onStream && typeof options.onStream === 'function') {
            // Implement streaming for text completion if needed
            // Similar to chatCompletion streaming implementation
        }

        try {
            const response = await fetch(`${NIM_CONFIG.ENDPOINTS.BASE_URL}${NIM_CONFIG.ENDPOINTS.COMPLETIONS}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
                },
                body: JSON.stringify({...params, stream: false}) // Force stream to false for regular requests
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error?.message || `API error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error("NIM API error:", error);
            throw error;
        }
    }
};

// Example send message function that supports streaming
async function sendChatMessage(userMessage, updateCallback) {
    // Show typing indicator or loading state
    if (typeof updateCallback === 'function') {
        updateCallback({ status: 'typing' });
    }

    try {
        // Get conversation history
        let messages = [];

        // Get chat history if function exists
        if (typeof getConversationHistory === 'function') {
            messages = getConversationHistory();
        }

        // Add the new user message
        messages.push({ role: "user", content: userMessage });

        // Prepare response handler for streaming
        let fullResponse = '';
        const handleStream = (data) => {
            fullResponse += data.text;
            if (typeof updateCallback === 'function') {
                updateCallback({
                    status: 'streaming',
                    message: fullResponse,
                    chunk: data.text
                });
            }
        };

        // Call the NIM API with streaming
        const response = await NimAPI.chatCompletion(messages, {
            stream: true,
            onStream: handleStream
        });

        // Final response is now in fullResponse
        if (typeof updateCallback === 'function') {
            updateCallback({
                status: 'complete',
                message: fullResponse
            });
        }

        return fullResponse;
    } catch (error) {
        console.error("Error sending message:", error);

        // Update the UI with the error
        if (typeof updateCallback === 'function') {
            updateCallback({
                status: 'error',
                error: error.message
            });
        }

        return null;
    }
}

// Define the sendMessage function globally to avoid reference errors
window.sendMessage = function() {
    console.log("Send button clicked");

    // Get the user input element
    const userInput = document.getElementById('user-input');
    if (!userInput) {
        console.error("User input element not found");
        return;
    }

    const messageText = userInput.value.trim();
    if (!messageText) {
        console.log("Empty message, not sending");
        return;
    }

    console.log("Sending message:", messageText);

    // Add user message to chat
    addMessageToChat(messageText, 'user');

    // Clear the input
    userInput.value = '';

    // Show typing indicator
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'block';
    }

    // Send the message to the backend
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            messages: [{ role: 'user', content: messageText }]
        })
    })
    .then(response => {
        console.log("Response status:", response.status);
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Response data:", data);

        // Hide typing indicator
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }

        // Add the bot's response to the chat
        if (data.status === 'success' && data.message) {
            addMessageToChat(data.message, 'bot');
        } else {
            addMessageToChat("Error: " + (data.message || "Unknown error"), 'bot');
        }
    })
    .catch(error => {
        console.error("Error:", error);

        // Hide typing indicator
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }

        // Show error message
        addMessageToChat(`Error: ${error.message}`, 'bot');
    });
};

// Helper function to add messages to chat
function addMessageToChat(text, sender) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error("Chat messages container not found");
        return;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = text;

    chatMessages.appendChild(messageDiv);

    // Scroll to the bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Also handle Enter key press
document.addEventListener('DOMContentLoaded', function() {
    const userInput = document.getElementById('user-input');
    if (userInput) {
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // Move all your JavaScript initialization code here

    // Example: Initialize event listeners only if elements exist
    document.getElementById('send-button')?.addEventListener('click', sendMessage);

    // Get the typing indicator element
    const typingIndicator = document.getElementById('typing-indicator');

    // Hide it initially
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }

    // Fix for connection status
    function updateConnectionStatus(status) {
      const connectionStatus = document.getElementById('connection-status');
      const connectionText = document.getElementById('connection-text');

      // Skip if elements don't exist
      if (!connectionStatus || !connectionText) return;

      switch (status) {
        case 'connecting':
          connectionStatus.style.backgroundColor = '#f59e0b';
          connectionText.textContent = 'Connecting...';
          break;
        // other cases...
      }
    }

    // Safe initialization of other elements
    if (typeof checkDocumentsStatus === 'function') {
      checkDocumentsStatus();
    }
});