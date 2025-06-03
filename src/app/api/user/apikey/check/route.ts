import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    // get user id from request headers (set by middleware)
    const userId = req.headers.get('x-user-id');
    if (!userId) {
      return NextResponse.json(
        {error: 'Unauthorized'},
        {status: 401}
      );
    }

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