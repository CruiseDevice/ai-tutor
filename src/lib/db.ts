// lib/db.ts
import { PrismaClient } from '@prisma/client'
import { withAccelerate } from '@prisma/extension-accelerate'

declare global {
  var prisma: PrismaClient | undefined
}

const prisma = global.prisma || new PrismaClient().$extends(withAccelerate())

if (process.env.NODE_ENV !== 'production') global.prisma = prisma

export default prisma