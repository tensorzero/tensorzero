#!/usr/bin/env node
/**
 * TensorZero Azure OpenAI Embeddings Example (Node.js)
 * 
 * This example demonstrates how to use Azure OpenAI embeddings through TensorZero
 * using the OpenAI SDK with TensorZero Gateway via HTTP.
 */

import OpenAI from 'openai';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function main() {
    console.log('☁️ TensorZero Azure OpenAI Embeddings Example (Node.js)');
    console.log('='.repeat(50));
    
    // Sample texts to embed
    const texts = [
        "Azure OpenAI provides enterprise-grade AI services.",
        "TensorZero seamlessly integrates with Azure OpenAI deployments.",
        "Enterprise AI applications require robust security and compliance."
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
        apiKey: process.env.AZURE_OPENAI_API_KEY || 'not-needed-for-gateway' // API key is handled by gateway
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
        
        // Batch text embeddings with different model
        console.log('🎯 Batch text embedding with larger model...');
        const batchResponse = await client.embeddings.create({
            input: texts,
            model: 'text-embedding-3-large'
        });
        
        console.log(`✅ Model: ${batchResponse.model}`);
        console.log(`📊 Generated ${batchResponse.data.length} embeddings`);
        console.log(`📏 Embedding dimensions: ${batchResponse.data[0].embedding.length}`);
        console.log(`🔢 Token usage - Prompt: ${batchResponse.usage.prompt_tokens}, Total: ${batchResponse.usage.total_tokens}`);
        console.log();
        
        // Try custom dimensions (if supported by deployment)
        console.log('🎛️ Custom dimensions example...');
        try {
            const customResponse = await client.embeddings.create({
                input: "Custom dimensions example",
                model: 'text-embedding-3-small',
                dimensions: 512
            });
            
            console.log(`📏 Requested 512 dimensions, got: ${customResponse.data[0].embedding.length}`);
            console.log();
        } catch (dimensionError) {
            console.log(`ℹ️ Custom dimensions not supported by this deployment: ${dimensionError.message}`);
            console.log();
        }
        
        // Show first few dimensions of an embedding
        const embedding = batchResponse.data[0].embedding;
        console.log(`🔍 First 5 dimensions of embedding: ${embedding.slice(0, 5).map(n => n.toFixed(6))}`);
        console.log();
        
        // Calculate embedding similarity
        const embedding1 = batchResponse.data[0].embedding;
        const embedding2 = batchResponse.data[1].embedding;
        const similarity = cosineSimilarity(embedding1, embedding2);
        console.log(`🎯 Cosine similarity between first two texts: ${similarity.toFixed(4)}`);
        console.log();
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.log('   Troubleshooting steps:');
        console.log('   1. Make sure TensorZero Gateway is running: curl http://localhost:3000/health');
        console.log('   2. Verify Azure OpenAI configuration in tensorzero.toml');
        console.log('   3. Check your Azure OpenAI API key is set: AZURE_OPENAI_API_KEY');
        console.log('   4. Ensure your Azure deployments exist and are active');
        console.log();
    }
    
    console.log('✨ Example completed!');
    console.log();
    console.log('💡 Tips:');
    console.log('   - Azure OpenAI deployments are region-specific');
    console.log('   - Update the endpoint and deployment_id in tensorzero.toml');
    console.log('   - Consider using different regions for redundancy');
    console.log('   - Monitor usage through Azure portal');
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

// Check for environment setup
if (!process.env.AZURE_OPENAI_API_KEY) {
    console.error('❌ Error: AZURE_OPENAI_API_KEY environment variable is required');
    console.log('   Please set your Azure OpenAI API key:');
    console.log('   export AZURE_OPENAI_API_KEY="your-api-key-here"');
    console.log('   Or create a .env file with: AZURE_OPENAI_API_KEY=your-api-key-here');
    console.log();
    console.log('   You can find your key in the Azure portal under:');
    console.log('   Your OpenAI Resource → Keys and Endpoint → Key 1');
    process.exit(1);
}

main().catch(console.error);