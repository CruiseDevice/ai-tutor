// lib/db.ts
import { PrismaClient } from '@prisma/client'

// prevent multiple instances of Prisma Client in development
declare global {
  const prisma: PrismaClient | undefined
}

const prisma = global.prisma || new PrismaClient()

if (process.env.NODE_ENV !== 'production') global.prisma = prisma

export default prisma