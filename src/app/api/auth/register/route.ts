// src/app/api/auth/register/route.ts
import prisma from "@/lib/db"
import { NextResponse } from "next/server"
import { z } from "zod"
import bcrypt from 'bcryptjs'
import { randomUUID } from 'crypto'

const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
})

export async function POST(req: Request) {
  try {
    console.log('Starting registration process...')
    
    const body = await req.json()
    console.log('Received body:', { ...body, password: '[REDACTED]' })
    
    const validation = registerSchema.safeParse(body)
    if(!validation.success) {
      console.log('Validation failed:', validation.error)
      return NextResponse.json(
        { error: 'Invalid input', details: validation.error.errors },
        { status: 400 }
      )
    }

    const { email, password } = validation.data
    console.log('Validated email:', email)

    console.log('Checking for existing user...')
    const existingUser = await prisma.user.findUnique({
      where: { email },
    })

    if(existingUser) {
      console.log('User already exists with email:', email)
      return NextResponse.json(
        { error: 'User already exists' },
        { status: 400 }
      )
    }

    console.log('Hashing password...')
    const hashedPassword = await bcrypt.hash(password, 10)
    console.log('Password hashed successfully')

    console.log('Creating new user...')
    const user = await prisma.user.create({
      data: {
        email,
        password: hashedPassword,
      }
    })
    console.log('User created:', { id: user.id, email: user.email })

    console.log('Creating session...')
    const session = await prisma.session.create({
      data: {
        userId: user.id,
        token: randomUUID(),
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
      }
    })
    console.log('Session created:', { id: session.id, userId: session.userId })

    const response = NextResponse.json(
      { message: 'Registration successful' },
      { status: 201 }
    )

    response.cookies.set('session_token', session.token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 30 * 24 * 60 * 60, // 30 days
      path: '/',
    })
    console.log('Cookie set and response prepared')

    return response

  } catch (error) {
    console.error('Registration error:', error instanceof Error ? error.message : String(error))
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    )
  }
}