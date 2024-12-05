const slider = document.getElementById('slider');
const beforeImg = document.getElementById('before');
const afterImg = document.getElementById('after');
const container = document.querySelector('.container');

// Create the split bar element and add it to the container
const splitBar = document.createElement('div');
splitBar.classList.add('split-bar');
container.appendChild(splitBar);

function updateSplitBar() {
    const value = slider.value;
    const percentage = (value / 100) * container.offsetWidth;
    splitBar.style.left = `${percentage - splitBar.offsetWidth / 2}px`; // Center the split bar on the slider value
    afterImg.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
}

// Initialize the split bar position
updateSplitBar();

// Update the split bar position when the slider value changes
slider.addEventListener('input', updateSplitBar);
