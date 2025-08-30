import React, { useState } from 'react'
import { RefreshCw, Clock, Settings, Search, ChevronDown, ChevronUp } from 'lucide-react'

const GenerationHistory = ({ generations, onRefresh, loading }) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(10)
  const [expandedGenerations, setExpandedGenerations] = useState(new Set())

  // Filter generations based on search term
  const filteredGenerations = generations.filter(gen =>
    gen.prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
    gen.response.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Pagination
  const totalPages = Math.ceil(filteredGenerations.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const currentGenerations = filteredGenerations.slice(startIndex, endIndex)

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const truncateText = (text, maxLength = 300) => {
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength) + '...'
  }

  const toggleExpanded = (generationId) => {
    setExpandedGenerations(prev => {
      const newSet = new Set(prev)
      if (newSet.has(generationId)) {
        newSet.delete(generationId)
      } else {
        newSet.add(generationId)
      }
      return newSet
    })
  }

  const handlePageChange = (page) => {
    setCurrentPage(page)
  }

  const handleRefresh = () => {
    setCurrentPage(1)
    setSearchTerm('')
    onRefresh()
  }

  if (generations.length === 0) {
    return (
      <div className="card text-center py-12">
        <Clock className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No generations yet</h3>
        <p className="text-gray-600 mb-4">
          Start by generating some text to see your history here.
        </p>
        <button
          onClick={handleRefresh}
          className="btn-primary"
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Generation History</h2>
          <p className="text-gray-600">
            {filteredGenerations.length} generation{filteredGenerations.length !== 1 ? 's' : ''} found
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          {/* Search */}
          <div className="relative">
            <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search generations..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-shakespeare-500 focus:border-transparent"
            />
          </div>
          
          {/* Refresh Button */}
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Generations List */}
      <div className="space-y-4">
        {currentGenerations.map((generation) => {
          const isExpanded = expandedGenerations.has(generation.id)
          const needsTruncation = generation.response.length > 300
          
          return (
            <div key={generation.id} className="card">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-shakespeare-600">
                    #{generation.id}
                  </span>
                  <span className="text-sm text-gray-500">
                    {formatDate(generation.created_at)}
                  </span>
                </div>
                
                {/* Generation Settings */}
                <div className="flex items-center space-x-2 text-xs text-gray-500">
                  <Settings className="h-3 w-3" />
                  <span>
                    T: {generation.temperature}, 
                    K: {generation.top_k || '∞'}, 
                    Tokens: {generation.max_new_tokens}
                  </span>
                </div>
              </div>

              {/* Prompt */}
              <div className="mb-3">
                <h4 className="text-sm font-medium text-gray-700 mb-1">Prompt:</h4>
                <p className="text-gray-900 bg-gray-50 p-3 rounded border border-gray-200">
                  {generation.prompt}
                </p>
              </div>

              {/* Generated Response */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-1">Generated Text:</h4>
                <div className="bg-blue-50 p-3 rounded border border-blue-200">
                  <p className="text-gray-800 font-serif leading-relaxed">
                    {isExpanded ? generation.response : truncateText(generation.response, 300)}
                  </p>
                  
                  {/* Show More/Less Toggle */}
                  {needsTruncation && (
                    <button
                      onClick={() => toggleExpanded(generation.id)}
                      className="mt-2 text-sm text-shakespeare-600 hover:text-shakespeare-700 font-medium flex items-center space-x-1"
                    >
                      {isExpanded ? (
                        <>
                          <ChevronUp className="h-4 w-4" />
                          <span>Show Less</span>
                        </>
                      ) : (
                        <>
                          <ChevronDown className="h-4 w-4" />
                          <span>Show More</span>
                        </>
                      )}
                    </button>
                  )}
                  
                  {/* Character Count */}
                  <div className="mt-2 text-xs text-gray-500">
                    {generation.response.length} characters generated
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center space-x-2">
          <button
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="btn-secondary px-3 py-1 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          
          {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
            <button
              key={page}
              onClick={() => handlePageChange(page)}
              className={`px-3 py-1 rounded ${
                currentPage === page
                  ? 'bg-shakespeare-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {page}
            </button>
          ))}
          
          <button
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="btn-secondary px-3 py-1 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}

      {/* No Results Message */}
      {filteredGenerations.length === 0 && searchTerm && (
        <div className="card text-center py-8">
          <Search className="h-8 w-8 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-600">No generations found matching "{searchTerm}"</p>
        </div>
      )}
    </div>
  )
}

export default GenerationHistory
