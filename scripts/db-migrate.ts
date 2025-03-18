// scripts/db-migrate.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

async function runMigration() {
  try {
    console.log('Starting database migration...');
    
    // Generate Prisma client
    console.log('Generating Prisma client...');
    await execAsync('npx prisma generate');
    
    // Run database migrations
    console.log('Running database migrations...');
    await execAsync('npx prisma db push');
    
    console.log('✅ Database migration completed successfully!');
  } catch (error) {
    console.error('❌ Database migration failed:', error);
    process.exit(1);
  }
}

runMigration();