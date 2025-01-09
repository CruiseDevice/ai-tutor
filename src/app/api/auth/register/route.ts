// /src/app/api/auth/register/route.ts

import { prisma } from "@/lib/prisma"
import { NextResponse } from "next/server"
import { z } from "zod"
import bcrypt from 'bcryptjs'

export async function GET() {
  try {
    const users = await prisma.user.findMany({})
    return NextResponse.json(users)
  } catch (error) {
    return NextResponse.json(
      {error: 'Error fetching users'},
      {status: 500}
    )
  }
}

// input validation schema
const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
})

export async function POST(req: Request) {
  try {
    const body = await req.json()

    // validate input
    const validation = registerSchema.safeParse(body);

    if(!validation.success) {
      return NextResponse.json(
        {error: 'Invalid input'},
        {status: 400}
      )
    }

    const {email, password} = validation.data

    // check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: {email},
    })

    if(existingUser) {
      return NextResponse.json(
        {error: 'User already exists'},
        {status: 400}
      )
    }

    // hash password
    const hashedPassword = await bcrypt.hash(password, 10)

    // create user
    const user = await prisma.user.create({
      data: {
        email,
        password: hashedPassword,
      }
    })

    // create sesssion
    const session = await prisma.session.create({
      data: {
        userId: user.id,
        token: crypto.randomUUID(),
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
      },
    })

    // set session cookie
    const response = NextResponse.json(
      {message: 'Registration successful'},
      {status: 201}
    )

    response.cookies.set('session_token', session.token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 30 * 24 * 60 * 60,  // 30 days
      path: '/'
    })

    return response

  } catch (error) {
    console.error('Registration error: ', error)
  }
}