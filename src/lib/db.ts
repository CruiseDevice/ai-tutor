// lib/db.ts
import { PrismaClient } from '@prisma/client'

// Define datasource options based on environment
const getDatabaseUrl = () => {
  if (process.env.NODE_ENV === 'production') {
    return process.env.PROD_DATABASE_URL || process.env.DATABASE_URL;
  }
  return process.env.DATABASE_URL;
}

// prevent multiple instances of Prisma Client in development
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

// Create Prisma client with dynamic URL
const prismaClientSingleton = () => {
  return new PrismaClient({
    datasources: {
      db: {
        url: getDatabaseUrl(),
      },
    },
  });
}

const prisma = globalForPrisma.prisma ?? prismaClientSingleton()

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma

export default prisma