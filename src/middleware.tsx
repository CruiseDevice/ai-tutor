import { NextRequest, NextResponse } from "next/server";
import { createClient } from '@vercel/edge-config';


// public routes that don't require authentication
const publicRoutes = [
  '/login', 
  '/register',
  '/api/auth/login',
  '/api/auth/register',
  '/api/auth/verify-session'
]

export async function middleware(request: NextRequest) {

  const path = request.nextUrl.pathname
  if(publicRoutes.some(route => path.startsWith(route))) {
    return NextResponse.next()
  }
  // get session token from cookies
  const sessionToken = request.cookies.get('session_token')?.value;

  if(!sessionToken) {
    return NextResponse.redirect(new URL('/login', request.url))
  }

  try {
    // find valid session
    const response = await fetch(`${request.nextUrl.origin}/api/auth/verify-session`, {
      headers: {
        'Cookie': `session_token=${sessionToken}`
      }
    })

    if (!response.ok) {
      // clear invalid session cookie and redirect to login
      const redirectResponse = NextResponse.redirect(new URL('/login', request.url))
      redirectResponse.cookies.delete('session_token')
      return redirectResponse
    }
    const session = await response.json()
    console.log(session)
    // add user information to request headers
    const requestHeaders = new Headers(request.headers)
    requestHeaders.set('x-user-id', session.userId)
    requestHeaders.set('x-user-email', session.userEmail)
    
    // return response with modified headers
    return NextResponse.next({
      request: {
        headers: requestHeaders,
      }
    })

  } catch(error) {
    console.error('Session verification error: ', error)
    return NextResponse.redirect(new URL('/login', request.url))
  }
}

export const config = {
  matcher: [
    // Match all paths except static files, api routes, and other public assets
    '/((?!_next/static|_next/image|favicon.ico|public|api/).*)',
  ]
}