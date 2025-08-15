#!/usr/bin/env node
/**
 * TensorZero OpenAI Embeddings Example (Node.js)
 * 
 * This example demonstrates how to use OpenAI embeddings through TensorZero
 * using the OpenAI SDK with TensorZero Gateway via HTTP.
 */

import OpenAI from 'openai';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function main() {
    console.log('🚀 TensorZero OpenAI Embeddings Example (Node.js)');
    console.log('='.repeat(50));
    
    // Sample texts to embed
    const texts = [
        "The quick brown fox jumps over the lazy dog.",
        "TensorZero is an observability and optimization platform for AI applications.",
        "Embeddings are dense vector representations of text that capture semantic meaning."
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
        apiKey: process.env.OPENAI_API_KEY || 'not-needed-for-gateway' // API key is handled by gateway
    });
    
    try {
        // Single text embedding
        console.log('🔍 Single text embedding...');
        const response = await client.embeddings.create({
            input: texts[0],
            model: 'text-embedding-3-small'
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
            model: 'text-embedding-3-large'
        });
        
        console.log(`✅ Model: ${batchResponse.model}`);
        console.log(`📊 Generated ${batchResponse.data.length} embeddings`);
        console.log(`📏 Embedding dimensions: ${batchResponse.data[0].embedding.length}`);
        console.log(`🔢 Token usage - Prompt: ${batchResponse.usage.prompt_tokens}, Total: ${batchResponse.usage.total_tokens}`);
        console.log();
        
        // Embedding with custom dimensions
        console.log('🎛️ Custom dimensions example...');
        const customResponse = await client.embeddings.create({
            input: "Custom dimensions example",
            model: 'text-embedding-3-small',
            dimensions: 512
        });
        
        console.log(`📏 Requested 512 dimensions, got: ${customResponse.data[0].embedding.length}`);
        console.log();
        
        // Show first few dimensions of an embedding
        const embedding = customResponse.data[0].embedding;
        console.log(`🔍 First 5 dimensions of embedding: ${embedding.slice(0, 5).map(n => n.toFixed(6))}`);
        console.log();
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.log('   Make sure TensorZero Gateway is running on port 3000');
        console.log('   Run: docker-compose up (if using Docker) or start the gateway locally');
        console.log();
    }
    
    console.log('✨ Example completed!');
    console.log();
    console.log('💡 Tips:');
    console.log('   - This example uses the TensorZero Gateway via HTTP');
    console.log('   - The gateway handles authentication and routing to OpenAI');
    console.log('   - Configure models in tensorzero.toml');
    console.log('   - Try different models by changing the "model" parameter');
}

// Check for environment setup
if (!process.env.OPENAI_API_KEY) {
    console.error('❌ Error: OPENAI_API_KEY environment variable is required');
    console.log('   Please set your OpenAI API key:');
    console.log('   export OPENAI_API_KEY="your-api-key-here"');
    console.log('   Or create a .env file with: OPENAI_API_KEY=your-api-key-here');
    process.exit(1);
}

main().catch(console.error);