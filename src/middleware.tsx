import { NextRequest, NextResponse } from "next/server";
import { createClient } from '@vercel/edge-config';


// routes that don't require authentication
const publicRoutes = [
  '/api/auth/login',
  '/api/auth/register',
  '/api/auth/verify-session'
]

const authRoutes = [
  '/login', 
  '/register',
]

const protectedRoutes = [
  '/dashboard',
  '/logout',
  '/Navbar',
]

export async function middleware(request: NextRequest) {

  // get the pathname
  const path = request.nextUrl.pathname

  if(publicRoutes.some(route => path.startsWith(route))) {
    return NextResponse.next()
  }
  // get session token from cookies
  const sessionToken = request.cookies.get('session_token')?.value;

  try {
    if (sessionToken){
      // find valid session

      const response = await fetch(`${request.nextUrl.origin}/api/auth/verify-session`, {
        headers: {
          'Cookie': `session_token=${sessionToken}`
        }
      })

      if (response.ok) {
        const session = await response.json()

        // if user is logged in and trying to access auth routes (login/register)
        // redirect them to dashboard
        if(authRoutes.some(route => path.startsWith(route))) {
          return NextResponse.redirect(new URL('/dashboard', request.url));
        }

        // add user information to request headers
        const requestHeaders = new Headers(request.headers)
        requestHeaders.set('x-user-id', session.userId)
        requestHeaders.set('x-user-email', session.userEmail)
        
        // return response with modified headers
        return NextResponse.next({
          request: {
            headers: requestHeaders,
          }
        });
      }
    }
    
    // invalid or no session
    // if trying to access protected routes, redirect to login
    if(protectedRoutes.some(route => path.startsWith(route))) {
      // clear invalid session cookie
      const response = NextResponse.redirect(new URL('/login', request.url));
      response.cookies.delete('session_token');
      return response;
    }

    // Allow access to auth routes for non-authenticated users
    if(authRoutes.some(route => path.startsWith(route))) {
      return NextResponse.next();
    }
    return NextResponse.next();

  } catch(error) {
    console.error('Session verification error: ', error)

    // on error, redirect to login for protected routes
    if(protectedRoutes.some(route => path.startsWith(route))) {
      return NextResponse.redirect(new URL('/login', request.url));
    }
    return NextResponse.next();
  }
}

export const config = {
  matcher: [
    // Match all paths except static files, api routes, and other public assets
    '/((?!_next/static|_next/image|favicon.ico|public|api/).*)',
  ]
}