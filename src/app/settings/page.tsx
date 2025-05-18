import APISettings from "@/components/APISettings";

export default function SettingsPage() {
  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gray-50">
      <main className="container mx-auto py-8 px-4">
        <h1 className="text-2xl font-bold mb-6">Settings</h1>
        <APISettings />
      </main>
    </div>
  )
}