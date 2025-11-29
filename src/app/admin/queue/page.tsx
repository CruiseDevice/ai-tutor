import { Metadata } from 'next';
import QueueDashboard from '@/components/admin/QueueDashboard';

export const metadata: Metadata = {
  title: 'Queue Monitor | Admin',
  description: 'Monitor document processing queue'
};

export default function QueueMonitorPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <QueueDashboard />
    </div>
  );
}
