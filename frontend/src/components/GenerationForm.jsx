import React, { useState } from 'react'
import { Send, Settings, Loader2 } from 'lucide-react'

const GenerationForm = ({ onGenerationComplete, setLoading, loading }) => {
  const [formData, setFormData] = useState({
    prompt: '',
    temperature: 1.0,
    top_k: 30,
    max_new_tokens: 100
  })
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [generatedText, setGeneratedText] = useState('')
  const [error, setError] = useState('')

  const handleInputChange = (e) => {
    const { name, value, type } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!formData.prompt.trim()) {
      setError('Please enter a prompt')
      return
    }

    setLoading(true)
    setError('')
    setGeneratedText('')

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setGeneratedText(data.response)
      onGenerationComplete(data)
      
      // Clear the prompt after successful generation
      setFormData(prev => ({ ...prev, prompt: '' }))
      
    } catch (error) {
      console.error('Generation failed:', error)
      setError('Failed to generate text. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const examplePrompts = [
    "To be, or not to be, that is the question:",
    "Now is the winter of our discontent",
    "All the world's a stage, and all the men and women merely players.",
    "The quality of mercy is not strained;",
    "Shall I compare thee to a summer's day?"
  ]

  const setExamplePrompt = (prompt) => {
    setFormData(prev => ({ ...prev, prompt }))
  }

  return (
    <div className="space-y-6">
      {/* Main Form */}
      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Prompt Input */}
          <div>
            <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
              Enter your prompt
            </label>
            <textarea
              id="prompt"
              name="prompt"
              value={formData.prompt}
              onChange={handleInputChange}
              rows={4}
              className="input-field resize-none"
              placeholder="Enter a prompt to generate Shakespeare-style text..."
              disabled={loading}
            />
          </div>

          {/* Advanced Settings Toggle */}
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-900"
            >
              <Settings className="h-4 w-4" />
              <span>Advanced Settings</span>
            </button>
          </div>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
              <div>
                <label htmlFor="temperature" className="block text-sm font-medium text-gray-700 mb-1">
                  Temperature
                </label>
                <input
                  type="number"
                  id="temperature"
                  name="temperature"
                  value={formData.temperature}
                  onChange={handleInputChange}
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  className="input-field"
                  disabled={loading}
                />
                <p className="text-xs text-gray-500 mt-1">0.1 = focused, 2.0 = creative</p>
              </div>

              <div>
                <label htmlFor="top_k" className="block text-sm font-medium text-gray-700 mb-1">
                  Top-K
                </label>
                <input
                  type="number"
                  id="top_k"
                  name="top_k"
                  value={formData.top_k}
                  onChange={handleInputChange}
                  min="1"
                  max="100"
                  className="input-field"
                  disabled={loading}
                />
                <p className="text-xs text-gray-500 mt-1">Limit token selection</p>
              </div>

              <div>
                <label htmlFor="max_new_tokens" className="block text-sm font-medium text-gray-700 mb-1">
                  Max Tokens
                </label>
                <input
                  type="number"
                  id="max_new_tokens"
                  name="max_new_tokens"
                  value={formData.max_new_tokens}
                  onChange={handleInputChange}
                  min="10"
                  max="500"
                  className="input-field"
                  disabled={loading}
                />
                <p className="text-xs text-gray-500 mt-1">Maximum new tokens to generate</p>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !formData.prompt.trim()}
            className="btn-primary w-full flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                <span>Generate Text</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Example Prompts */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Example Prompts</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {examplePrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => setExamplePrompt(prompt)}
              className="text-left p-3 text-sm text-gray-700 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors duration-200"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      {/* Generated Text Display */}
      {generatedText && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Generated Text</h3>
          <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
            <p className="text-gray-800 font-serif leading-relaxed whitespace-pre-wrap">
              {generatedText}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default GenerationForm
