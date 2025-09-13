import { Badge } from "@/components/ui/badge"

export function CodingAssignmentsSection() {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-50 to-violet-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">📝 Coding Assignments</h3>
        <p className="text-gray-700 mb-4">
          Collaborative coding assignments with real-time feedback and automated testing.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Coding</Badge>
          <Badge variant="outline">Assignments</Badge>
          <Badge variant="outline">Feedback</Badge>
        </div>
      </div>
      <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
        <h4 className="font-semibold text-yellow-900 mb-2">🚧 Coming Soon</h4>
        <p className="text-yellow-800">These coding assignments are currently under development.</p>
      </div>
    </div>
  )
}
