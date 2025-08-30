import React, { useState, useEffect } from 'react'
import GenerationForm from './components/GenerationForm'
import GenerationHistory from './components/GenerationHistory'
import { BookOpen, Sparkles, History } from 'lucide-react'

function App() {
  const [generations, setGenerations] = useState([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('generate')

  const addGeneration = (newGeneration) => {
    setGenerations(prev => [newGeneration, ...prev])
  }

  const loadHistory = async () => {
    try {
      const response = await fetch('/api/history?limit=50')
      const data = await response.json()
      setGenerations(data)
    } catch (error) {
      console.error('Failed to load history:', error)
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-shakespeare-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-shakespeare-600 p-2 rounded-lg">
                <BookOpen className="h-6 w-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">
                ShakespeareGPT
              </h1>
            </div>
            <p className="text-sm text-gray-600">
              AI-Powered Shakespeare-Style Text Generation
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-white p-1 rounded-lg shadow-sm border border-gray-200 mb-8">
          <button
            onClick={() => setActiveTab('generate')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors duration-200 ${
              activeTab === 'generate'
                ? 'bg-shakespeare-100 text-shakespeare-700'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            <Sparkles className="h-4 w-4" />
            <span>Generate Text</span>
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors duration-200 ${
              activeTab === 'history'
                ? 'bg-shakespeare-100 text-shakespeare-700'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            <History className="h-4 w-4" />
            <span>Generation History</span>
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'generate' ? (
          <GenerationForm 
            onGenerationComplete={addGeneration}
            setLoading={setLoading}
            loading={loading}
          />
        ) : (
          <GenerationHistory 
            generations={generations}
            onRefresh={loadHistory}
            loading={loading}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>
              Powered by a custom-trained language model • Built with FastAPI & React
            </p>
            <p className="mt-1">
              Generate Shakespeare-style text with AI assistance
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
