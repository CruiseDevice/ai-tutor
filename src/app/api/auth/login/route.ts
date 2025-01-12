// src/app/api/auth/login/route.ts
import { NextResponse } from "next/server"
import { z } from "zod"
import bcrypt from 'bcryptjs'
import prisma from "@/lib/db"
import { randomUUID } from "crypto"

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string(),
})

export async function POST(req: Request) {
  try {
    console.log('Starting login process...')
    const body = await req.json()
    
    // Validate input
    const validation = loginSchema.safeParse(body)
    if(!validation.success) {
      console.log('Validation failed:', validation.error)
      return NextResponse.json(
        { error: 'Invalid input', details: validation.error.errors },
        { status: 400 }
      )
    }

    const { email, password } = validation.data
    console.log('Attempting login for email:', email)

    // Find user
    const user = await prisma.user.findUnique({
      where: { email },
    })

    if (!user) {
      console.log('User not found')
      return NextResponse.json(
        { error: 'Invalid credentials' },
        { status: 401 }
      )
    }

    // Verify password
    const passwordValid = await bcrypt.compare(password, user.password)
    
    if(!passwordValid) {
      console.log('Invalid password')
      return NextResponse.json(
        { error: 'Invalid credentials' },
        { status: 401 }
      )
    }

    console.log('Password verified, creating session...')

    // Create new session
    const session = await prisma.session.create({
      data: {
        userId: user.id,
        token: randomUUID(),
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days
      },
    })

    console.log('Session created successfully')

    // Set session cookie and return response
    const response = NextResponse.json(
      { message: 'Login successful' },
      { status: 200 }
    )

    // set session cookie
    response.cookies.set({
      name:'session_token',
      value: session.token,
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      expires: session.expiresAt,
      path: '/',
    })

    console.log('Response prepared with session cookie')
    return response

  } catch (error) {
    console.error('Login error:', error instanceof Error ? error.message : String(error))
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    )
  }
}