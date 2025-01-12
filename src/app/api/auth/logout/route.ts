// /src/app/api/auth/logout/route.ts
import { NextResponse } from "next/server";
import prisma from '@/lib/db'

export async function POST(req: Request) {
  try{
    // get the session token from cookies
    const cookieHeader = req.headers.get('cookie');
    const sessionToken = cookieHeader?.split(';')
      .find(cookie => cookie.trim().startsWith('session_token='))
      ?.split('=')[1];

    if(sessionToken) {
      // delete the session from the database
      await prisma?.session.deleteMany({
        where: {
          token: sessionToken
        }
      });
    }

    // create response with cleared cookie
    const response = NextResponse.json(
      {message: 'Logged out successfully'},
      {status: 200}
    );

    // clear the session cookie
    response.cookies.delete('session_token');
    return response;
  } catch (error) {
    console.error('Logout error: ', error);
    return NextResponse.json(
      {error: 'Internal server error'},
      {status: 500}
    )
  }
}