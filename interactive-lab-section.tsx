import { Badge } from "@/components/ui/badge"

export function InteractiveLabSection() {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-green-50 to-teal-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎛️ Interactive ML Lab</h3>
        <p className="text-gray-700 mb-4">Collection of interactive dashboards and tools for hands-on ML learning.</p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Interactive</Badge>
          <Badge variant="outline">Dashboards</Badge>
          <Badge variant="outline">Hands-on</Badge>
        </div>
      </div>
      <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
        <h4 className="font-semibold text-yellow-900 mb-2">🚧 Coming Soon</h4>
        <p className="text-yellow-800">This interactive lab is currently under development.</p>
      </div>
    </div>
  )
}
