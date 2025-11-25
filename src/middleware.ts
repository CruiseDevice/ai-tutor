// src/middleware.tsx
import { NextRequest, NextResponse } from "next/server";

// routes that don't require authentication
const publicRoutes = [
  '/login',
  '/register',
  '/forgot-password',
  '/reset-password',
  '/',
]

export async function middleware(request: NextRequest) {
  // get the pathname
  const path = request.nextUrl.pathname

  // Allow access to public routes
  if(publicRoutes.some(route => path === route || path.startsWith(route))) {
    return NextResponse.next()
  }

  // get session token from cookies
  const sessionToken = request.cookies.get('session_token')?.value;

  // If no session token and trying to access protected route, redirect to login
  if (!sessionToken) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // If session token exists, allow access (verification happens on API calls)
  return NextResponse.next();
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