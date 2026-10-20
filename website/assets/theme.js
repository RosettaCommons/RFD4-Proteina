// Share explicit theme choices between the home page and benchmark explorer.
(()=>{
  const root=document.documentElement,key='rfd4-proteina-theme';
  try{const saved=localStorage.getItem(key);if(['light','dark'].includes(saved))root.dataset.theme=saved}catch{}
  function label(){const button=document.getElementById('theme');if(button){const next=root.dataset.theme==='dark'?'Light':'Dark';button.textContent=next+' mode';button.setAttribute('aria-label','Switch to '+next.toLowerCase()+' mode')}}
  document.addEventListener('DOMContentLoaded',label);
  document.addEventListener('click',event=>{
    if(!event.target.closest('#theme'))return;
    root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';
    try{localStorage.setItem(key,root.dataset.theme)}catch{}
    label();window.dispatchEvent(new Event('site-theme-change'));
  });
})();
