// app/api/user/apikey/route.ts

import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    // get userID from request headers (set by middleware)
    const userId = req.headers.get('x-user-id');
    if(!userId) {
      return NextResponse.json(
        {error: 'Unauthorized'},
        {status: 401}
      )
    };

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
    console.error('Error updating API key: ', error);
    return NextResponse.json(
      {error: 'Failed to update API key'},
      {status: 500}
    )
  }
}