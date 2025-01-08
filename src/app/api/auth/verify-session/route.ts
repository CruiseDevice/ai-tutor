// app/api/auth/verify-session/route.ts
import { NextResponse } from 'next/server'
import prisma from '@/lib/db'

export async function GET(request: Request) {
  try {
    const sessionToken = request.headers.get('cookie')
      ?.split(';')
      ?.find(c => c.trim().startsWith('session_token='))
      ?.split('=')[1]

    if (!sessionToken) {
      return NextResponse.json({ error: 'No session token' }, { status: 401 })
    }

    const session = await prisma.session.findFirst({
      where: {
        token: sessionToken,
        expiresAt: {
          gt: new Date(),
        },
      },
      include: {
        user: true,
      },
    })

    if (!session) {
      return NextResponse.json({ error: 'Invalid session' }, { status: 401 })
    }
    return NextResponse.json({
      userId: session.userId,
      userEmail: session.user.email
    })

  } catch (error) {
    console.error('Session verification error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}