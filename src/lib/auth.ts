import { cookies } from "next/headers";
import prisma from "./db";

export interface AuthUser {
  id: string;
  email: string;
}

export async function auth(): Promise<AuthUser | null> {
  try{
    // get the session token from cookies
    const cookieStore = cookies();
    const sessionToken = (await cookieStore).get('session_token')?.value;

    if(!sessionToken) {
      return null;
    }

    // find the session in the database
    const session = await prisma.session.findUnique({
      where: {
        token: sessionToken
      },
      include: {
        user: {
          select: {
            id: true,
            email: true
          }
        }
      }
    });

    // check if session exists and is not expired
    if(!session || session.expiresAt < new Date()) {
      return null;
    }

    // return user information
    return {
      id: session.user.id,
      email: session.user.email
    };
  } catch (error) {
    console.error('Auth error: ', error);
    return null;
  }
}