// app/api/conversations/route.ts

import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    const user = await auth();
    if (!user) {
      return NextResponse.json(
        {error: "Unauthorized"},
        {status: 401}
      );
    }

    // fetch all conversations for the user with document info
    const conversations = await prisma.conversation.findMany({
      where: {
        userId: user.id,
      },
      include: {
        document: {
          select: {
            title: true,
          },
        },
      },
      orderBy: {
        updatedAt: 'desc'
      },
    });

    // transform the data for the frontend
    const transformedConversations = conversations.map(conversation => ({
      id: conversation.id,
      documentId: conversation.documentId,
      title: conversation.document.title,
      updatedAt: conversation.updatedAt,
    }));

    return NextResponse.json({
      conversations: transformedConversations,
    });
  } catch (error) {
    console.error("Error fetching conversations: ", error);
    return NextResponse.json(
      {error: "Failed to fetch conversations"},
      {status: 500}
    )
  }
}