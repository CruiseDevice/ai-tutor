// app/api/auth/user/route.ts
import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function GET() {
  try {
    // get the session token from cookies
    const cookieStore = cookies();
    const sessionToken = (await cookieStore).get('session_token')?.value;

    if (!sessionToken) {
      return new Response('No session token', {
        status: 401,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }

    // find the session and associated user
    const session = await prisma?.session.findUnique({
      where: {token: sessionToken},
      include: {user: true}
    });

    if(!session) {
      return new Response('Session not found', {
        status: 401,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }

    if(session.expiresAt < new Date()) {
      return new Response('Session expired', {
        status: 401,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }

    // return the user id
    return NextResponse.json({
      id: session.userId,
      email: session.user.email
    });
  } catch (error) {
    console.error('Error getting user: ', error);
    return new Response('Internal server error', {
      status: 500,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
}