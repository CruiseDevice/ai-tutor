// app/api/user/apikey/route.ts

import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    // get session token from cookies
    const sessionToken = req.cookies.get('session_token')?.value;

    if (!sessionToken) {
      return NextResponse.json(
        {error: 'Unauthorized - No session token'},
        {status: 401}
      )
    }

    // verify session and get user ID
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
      return NextResponse.json(
        {error: 'Unauthorized - Invalid session'},
        {status: 401}
      )
    }

    const userId = session.userId;

    const {apiKey} = await req.json();
    if(!apiKey || !apiKey.startsWith('sk-')) {
      return NextResponse.json(
        {error: 'Invalid API key format'},
        {status: 400}
      );
    }

    // update user's API key
    await prisma.user.update({
      where: {id: userId},
      data: {apiKey}
    })
    return NextResponse.json({ message: 'API key updated successfully' });
  } catch (error) {
    return NextResponse.json(
      {error: error},
      {status: 500}
    )
  }
}