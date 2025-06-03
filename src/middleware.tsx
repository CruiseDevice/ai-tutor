// src/middleware.tsx
import { NextRequest, NextResponse } from "next/server";

// routes that don't require authentication
const publicRoutes = [
  '/api/auth/login',
  '/api/auth/register',
  '/api/auth/verify-session',
  '/api/auth/forgot-password',
  '/api/auth/password-reset/request',
  '/api/auth/password-reset/confirm',
  '/forgot-password',
  '/reset-password',
  '/',
]

export async function middleware(request: NextRequest) {

  // get the pathname
  const path = request.nextUrl.pathname

  if(publicRoutes.some(route => path.startsWith(route))) {
    return NextResponse.next()
  }

  // get session token from cookies
  const sessionToken = request.cookies.get('session_token')?.value;

  // check if it is an auth route (login/register)
  const isAuthRoute = path === '/login' || path === '/register';

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
        if(isAuthRoute) {
          return NextResponse.redirect(new URL('/dashboard', request.url));
        }

        // add user information to request headers
        const requestHeaders = new Headers(request.headers)
        requestHeaders.set('x-user-id', session.userId)
        requestHeaders.set('x-user-email', session.userEmail)
        
        // return response with modified headers
        const newResponse = NextResponse.next({
          request: {
            headers: requestHeaders,
          }
        });
        // Also set the headers in the response
        newResponse.headers.set('x-user-id', session.userId);
        newResponse.headers.set('x-user-email', session.userEmail);
        return newResponse
      }
    }

    // no valid session
    if (!isAuthRoute && path !== '/login') {
      // redirect to login for protected routes
      return NextResponse.redirect(new URL('/login', request.url));
    }

    return NextResponse.next();
  } catch(error) {
    console.error('Middleware error: ', error)

    if(!isAuthRoute && path !== '/') {
      return NextResponse.redirect(new URL('/login', request.url))
    }

    return NextResponse.next();
  }
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public (public files)
     */
    '/((?!_next/static|_next/image|favicon.ico|public/).*)',
  ],
}