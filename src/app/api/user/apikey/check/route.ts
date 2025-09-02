import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
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

    // Get user's API key status
    const user = await prisma.user.findUnique({
      where: {id: userId},
      select:{apiKey: true}
    });

    return NextResponse.json({
      hasApiKey: !!user?.apiKey,
    });
  } catch (error) {
    return NextResponse.json(
      {error: 'Failed to check API key status'},
      {status: 500}
    );
  }
}