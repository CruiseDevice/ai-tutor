import { NextResponse } from "next/server"
import { z } from "zod"
import bcrypt from 'bcryptjs'
import { PrismaClient } from "@prisma/client"

const prisma = new PrismaClient

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string(),
})

export async function POST(req: Request) {
  try {
    const body = await req.json()

    // validate input
    const validation = loginSchema.safeParse(body)
    if(!validation.success) {
      return NextResponse.json(
        {error: 'Invalid input'},
        {status: 400}
      )
    }

    const {email, password} = validation.data

    // find user
    const user = await prisma.user.findUnique({
      where: {email},
    })

    if (!user) {
      return NextResponse.json(
        {error: 'Invalid credentials'},
        {status: 401}
      )
    }

    // verify password
    const passwordValid = await bcrypt.compare(password, user.password)
    
    if(!passwordValid) {
      return NextResponse.json(
        {error: 'Invalid credentials'},
        {status: 401}
      )
    }

    // Create new session
    const session = await prisma.session.create({
      data: {
        userId: user.id,
        token: crypto.randomUUID(),
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
      },
    })

    // Set session cookie
    const response = NextResponse.json({ message: 'Login successful' })
    response.cookies.set('session_token', session.token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 30 * 24 * 60 * 60, // 30 days
      path: '/',
    })

    return response


  } catch (error) {
    console.error('Login error', error)
    return NextResponse.json(
      {error: 'Internal server error'},
      {status: 500}
    )
  }
}