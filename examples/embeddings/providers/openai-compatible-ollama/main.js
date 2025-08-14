#!/usr/bin/env node
/**
 * TensorZero Ollama Embeddings Example (Node.js)
 * 
 * This example demonstrates how to use Ollama embeddings through TensorZero
 * using the OpenAI SDK with TensorZero Gateway via HTTP.
 */

import OpenAI from 'openai';

async function main() {
    console.log('🦙 TensorZero Ollama Embeddings Example (Node.js)');
    console.log('='.repeat(50));
    
    // Sample texts to embed
    const texts = [
        "Local embeddings with Ollama are fast and private.",
        "TensorZero provides a unified interface for multiple embedding providers.",
        "Nomic Embed Text is a high-quality embedding model that runs locally."
    ];
    
    console.log('📝 Input texts:');
    texts.forEach((text, i) => {
        console.log(`  ${i + 1}. ${text}`);
    });
    console.log();
    
    // Using OpenAI SDK with TensorZero Gateway via HTTP
    console.log('OpenAI SDK with TensorZero Gateway (HTTP)');
    console.log('-'.repeat(40));
    
    // Initialize OpenAI client pointing to TensorZero Gateway
    const client = new OpenAI({
        baseURL: 'http://localhost:3000/openai/v1',
        apiKey: 'not-needed' // No API key required for local Ollama
    });
    
    try {
        // Single text embedding
        console.log('🔍 Single text embedding...');
        const response = await client.embeddings.create({
            input: texts[0],
            model: 'nomic-embed-text'
        });
        
        console.log(`✅ Model: ${response.model}`);
        console.log(`📊 Generated ${response.data.length} embedding(s)`);
        console.log(`📏 Embedding dimensions: ${response.data[0].embedding.length}`);
        console.log(`🔢 Token usage - Prompt: ${response.usage.prompt_tokens}, Total: ${response.usage.total_tokens}`);
        console.log();
        
        // Batch text embeddings
        console.log('🎯 Batch text embedding...');
        const batchResponse = await client.embeddings.create({
            input: texts,
            model: 'nomic-embed-text'
        });
        
        console.log(`✅ Model: ${batchResponse.model}`);
        console.log(`📊 Generated ${batchResponse.data.length} embeddings`);
        console.log(`📏 Embedding dimensions: ${batchResponse.data[0].embedding.length}`);
        console.log(`🔢 Token usage - Prompt: ${batchResponse.usage.prompt_tokens}, Total: ${batchResponse.usage.total_tokens}`);
        console.log();
        
        // Show first few dimensions of an embedding
        const embedding = batchResponse.data[0].embedding;
        console.log(`🔍 First 5 dimensions of embedding: ${embedding.slice(0, 5).map(n => n.toFixed(6))}`);
        console.log();
        
        // Calculate similarity between two embeddings
        const embedding1 = batchResponse.data[0].embedding;
        const embedding2 = batchResponse.data[1].embedding;
        const similarity = cosineSimilarity(embedding1, embedding2);
        console.log(`🎯 Cosine similarity between first two texts: ${similarity.toFixed(4)}`);
        console.log();
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.log('   Make sure TensorZero Gateway and Ollama are running:');
        console.log('   docker-compose up');
        console.log();
        console.log('   Check service status:');
        console.log('   curl http://localhost:3000/health  # Gateway');
        console.log('   curl http://localhost:11434/api/tags  # Ollama');
        console.log();
    }
    
    console.log('✨ Example completed!');
    console.log();
    console.log('💡 Tips:');
    console.log('   - Ollama embeddings run locally and don\'t require API keys');
    console.log('   - nomic-embed-text produces 768-dimensional embeddings');
    console.log('   - You can try other embedding models by pulling them with Ollama');
    console.log('   - Use docker-compose logs ollama to see Ollama\'s output');
}

/**
 * Calculate cosine similarity between two embeddings
 */
function cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

main().catch(console.error);