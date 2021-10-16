const stats = require("./stats")
// @ponicode
describe("stats.toggleStats", () => {
    test("0", () => {
        let callFunction = () => {
            stats.toggleStats()
        }
    
        expect(callFunction).not.toThrow()
    })
})

// @ponicode
describe("stats.toggleTotalCount", () => {
    test("0", () => {
        let callFunction = () => {
            stats.toggleTotalCount()
        }
    
        expect(callFunction).not.toThrow()
    })
})
