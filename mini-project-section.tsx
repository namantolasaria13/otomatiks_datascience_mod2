import { Badge } from "@/components/ui/badge"

export function MiniProjectSection() {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">💻 Mini Project: Boston Housing</h3>
        <p className="text-gray-700 mb-4">
          Complete end-to-end machine learning project for predicting housing prices.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Regression</Badge>
          <Badge variant="outline">End-to-End</Badge>
          <Badge variant="outline">Real Dataset</Badge>
        </div>
      </div>
      <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
        <h4 className="font-semibold text-yellow-900 mb-2">🚧 Coming Soon</h4>
        <p className="text-yellow-800">This mini project is currently under development.</p>
      </div>
    </div>
  )
}
