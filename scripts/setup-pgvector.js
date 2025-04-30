// scripts/setup-pgvector.js
import { exec } from 'child_process';
import { promisify } from 'util';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

const execAsync = promisify(exec);

// Get the directory name in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function setupPgvector() {
  try {
    console.log('🚀 Setting up pgvector extension...');
    
    // Step 1: Generate Prisma client
    console.log('📦 Generating Prisma client...');
    await execAsync('npx prisma generate');
    
    // Step 2: Push the schema to the database
    console.log('🗃️ Pushing schema to database...');
    await execAsync('npx prisma db push');
    
    // Step 3: Run the pgvector setup SQL using Docker
    console.log('🔧 Setting up pgvector extension...');
    const sqlFilePath = path.join(__dirname, '../prisma/migrations/pgvector_setup.sql');
    const sqlContent = await fs.readFile(sqlFilePath, 'utf8');
    
    // Execute SQL commands using Docker
    const cmd = `docker exec study-fetch-tutor-db-1 psql -U postgres -d studyfetch -c "${sqlContent.replace(/"/g, '\\"')}"`;
    await execAsync(cmd);
    
    console.log('✅ pgvector setup completed successfully!');
  } catch (error) {
    console.error('❌ pgvector setup failed:', error);
    process.exit(1);
  }
}

setupPgvector();