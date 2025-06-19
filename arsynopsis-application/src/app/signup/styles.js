// selectStyles.js
const selectStyles = {
  control: (base, state) => ({
    ...base,
    minHeight: 40, // match input height
    height: 40,
    padding: "0 10px", // horizontal padding to match input's p-2.5 (10px)
    borderRadius: 8, // rounded-lg = 0.5rem = 8px
    backgroundColor: "#374151", // dark:bg-gray-700
    borderColor: state.isFocused ? "#3b82f6" : "#4b5563",
    boxShadow: state.isFocused ? "0 0 0 1px #3b82f6" : "none",
    color: "white",
    "&:hover": {
      borderColor: "#3b82f6",
    },
  }),
  valueContainer: (base) => ({
    ...base,
    padding: "2px 6px", // reduce padding inside control to align text vertically
    height: 36, // slightly less than control to center text
  }),
  input: (base) => ({
    ...base,
    margin: 0,
    padding: 0,
    color: "white",
    fontSize: 14, // text-sm = 14px
    lineHeight: "20px",
  }),
  multiValue: (base) => ({
    ...base,
    backgroundColor: "#4b5563", // dark:border-gray-600
    height: 28, // smaller height for tags to fit input size
  }),
  multiValueLabel: (base) => ({
    ...base,
    color: "white",
    fontSize: 14,
    lineHeight: "20px",
    padding: "3px 6px",
  }),
  multiValueRemove: (base) => ({
    ...base,
    color: "white",
    ":hover": {
      backgroundColor: "#3b82f6",
      color: "white",
    },
  }),
  indicatorsContainer: (base) => ({
    ...base,
    height: 40, // match control height
  }),
  placeholder: (base) => ({
    ...base,
    color: "#9ca3af", // dark:placeholder-gray-400
    fontSize: 14,
  }),
  singleValue: (base) => ({
    ...base,
    color: "white",
    fontSize: 14,
  }),
  menu: (base) => ({
    ...base,
    backgroundColor: "#374151",
    color: "white",
  }),
  option: (base, state) => ({
    ...base,
    backgroundColor: state.isFocused ? "#4b5563" : "#374151",
    color: "white",
    cursor: "pointer",
  }),
};

export default selectStyles;
